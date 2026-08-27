from __future__ import annotations
import os
os.environ.setdefault('MKL_THREADING_LAYER', 'GNU')
import argparse
import sys
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.utils.data import DataLoader
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from models.ionet_a import build_ionet_a
from scripts.experiment_utils import load_scalers
from scripts.metrics_probabilistic import coverage
from scripts.npz_dataset import NPZDataset, load_meta, split_npz_paths
from scripts.probabilistic_eval import build_e2_probabilistic_metrics, persistence_pred_obs, save_probabilistic_metrics
MODEL_NAME = 'ionet_a'

@torch.no_grad()
def predict_ionet_a_quantiles(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    fc_q, fc_med = ([], [])
    fc_y, fc_m = ([], [])
    q_levels: np.ndarray | None = None
    for batch in loader:
        out = model(batch['x'].to(device), batch['mask'].to(device), None, soft_y=batch['soft_y'].to(device), soft_y_mask=batch['soft_y_mask'].to(device), soft_lab_mask=batch['soft_lab_mask'].to(device) if 'soft_lab_mask' in batch else None)
        fq = out['forecast_quantiles'].cpu().numpy()
        fc_q.append(fq)
        fc_med.append(out['forecast'].cpu().numpy())
        if q_levels is None:
            q_levels = out['quantile_levels'].cpu().numpy()
        fc_y.append(batch['forecast_y'].numpy())
        fc_m.append(batch['forecast_y_mask'].numpy())
    pred = {'forecast_quantiles': np.concatenate(fc_q, 0), 'forecast': np.concatenate(fc_med, 0), 'quantile_levels': q_levels if q_levels is not None else np.array([0.1, 0.5, 0.9])}
    obs = {'forecast': np.concatenate(fc_y, 0), 'forecast_mask': np.concatenate(fc_m, 0)}
    return (pred, obs)

def find_interval_scale(pred_q: np.ndarray, obs: np.ndarray, mask: np.ndarray, q_levels: np.ndarray, target_cov: float=0.9) -> float:
    levels = np.asarray(q_levels, dtype=np.float64)
    med_idx = int(np.argmin(np.abs(levels - 0.5)))
    lo_idx, hi_idx = (0, len(levels) - 1)
    best_s, best_gap = (1.0, 1.0)
    for s in np.linspace(1.0, 4.0, 31):
        med = pred_q[med_idx]
        q_lo = med - s * (med - pred_q[lo_idx])
        q_hi = med + s * (pred_q[hi_idx] - med)
        cov = coverage(q_lo, q_hi, obs, mask)
        gap = abs(cov - target_cov)
        if gap < best_gap:
            best_gap, best_s = (gap, float(s))
    return best_s

def apply_interval_scale(pred: dict[str, np.ndarray], scale: float) -> dict[str, np.ndarray]:
    if scale <= 1.0 + 1e-06:
        return pred
    fq = pred['forecast_quantiles'].copy()
    q_levels = pred['quantile_levels']
    med_idx = int(np.argmin(np.abs(q_levels - 0.5)))
    lo_idx, hi_idx = (0, len(q_levels) - 1)
    med = fq[:, med_idx]
    fq[:, lo_idx] = med - scale * (med - fq[:, lo_idx])
    fq[:, hi_idx] = med + scale * (fq[:, hi_idx] - med)
    out = dict(pred)
    out['forecast_quantiles'] = fq
    out['forecast'] = fq[:, med_idx]
    out['interval_scale'] = np.array([scale], dtype=np.float32)
    return out

def pick_device(name: str) -> torch.device:
    return torch.device('cuda' if name == 'cuda' and torch.cuda.is_available() else 'cpu')

def _load_ionet_a(ckpt_path: Path, meta: dict[str, Any], device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if ckpt.get('model') not in ('ionet_a', 'ionet_lite', None):
        if ckpt.get('model') == 'ionet':
            raise SystemExit(f'Checkpoint is IO-Net v1.0 ({ckpt_path}); need IO-Net-A. Retrain or remove stale best.pt.')
        raise SystemExit(f"Unexpected checkpoint model={ckpt.get('model')!r} in {ckpt_path}")
    cfg = ckpt.get('ionet_a_cfg') or ckpt.get('ionet_lite_cfg') or {}
    kw: dict[str, Any] = dict(in_channels=len(meta['easy_vars']), seq_len=int(meta['seq_len']), pred_len=int(meta['pred_len']), n_forecast=len(meta['forecast_vars']))
    for k in ('hidden', 'dropout', 'quantiles', 'use_persist_anchor', 'use_diurnal_template', 'use_mask_gate', 'use_lab_age', 'use_monotonic_quantiles', 'width_reg_weight'):
        if k in cfg:
            kw[k] = cfg[k]
    model = build_ionet_a(**kw)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.to(device)
    model.eval()
    return model

def main() -> None:
    ap = argparse.ArgumentParser(description='Evaluate IO-Net-A')
    ap.add_argument('--proc-dir', type=Path, required=True)
    ap.add_argument('--results-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--split', default='test')
    ap.add_argument('--denormalize', type=int, default=1)
    ap.add_argument('--horizons', default='1,2,3,6')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--min-valid-n', type=int, default=30)
    ap.add_argument('--calibrate', type=int, default=0, help='1=optional val interval widening (supplement only; main text uses 0)')
    args = ap.parse_args()
    meta = load_meta(args.proc_dir)
    device = pick_device(args.device)
    ckpt = args.results_dir / 'best.pt'
    if not ckpt.exists():
        raise SystemExit(f'Missing checkpoint: {ckpt}')
    model = _load_ionet_a(ckpt, meta, device)
    paths = split_npz_paths(args.proc_dir)
    horizons = tuple((int(h) for h in args.horizons.split(',') if h.strip()))
    scalers = load_scalers(args.proc_dir / 'scalers.json') if args.denormalize else None
    pred, obs = predict_ionet_a_quantiles(model, DataLoader(NPZDataset(paths[args.split]), batch_size=args.batch_size), device)
    interval_scale = 1.0
    if args.calibrate:
        val_pred, val_obs = predict_ionet_a_quantiles(model, DataLoader(NPZDataset(paths['val']), batch_size=args.batch_size), device)
        chla_i = meta['forecast_vars'].index('chla') if 'chla' in meta['forecast_vars'] else 1
        interval_scale = find_interval_scale(val_pred['forecast_quantiles'][:, :, chla_i, 5].T, val_obs['forecast'][:, chla_i, 5], val_obs['forecast_mask'][:, chla_i, 5], val_pred['quantile_levels'])
        pred = apply_interval_scale(pred, interval_scale)
    p_pred, _ = persistence_pred_obs(args.proc_dir, args.split, args.batch_size)
    persist_met = build_e2_probabilistic_metrics(p_pred, obs, meta, scalers, bool(args.denormalize), horizons, args.min_valid_n)
    met = build_e2_probabilistic_metrics(pred, obs, meta, scalers, bool(args.denormalize), horizons, args.min_valid_n, pred['quantile_levels'], persist_met)
    out = args.out or args.results_dir / 'E2_crps.json'
    save_probabilistic_metrics(out, MODEL_NAME, met, extra={'quantile_levels': pred['quantile_levels'].tolist(), 'interval_scale': interval_scale})
    chla = met.get('by_target', {}).get('chla', {}).get('h6', {})
    print(f"[evaluate_{MODEL_NAME}] saved -> {out}\n  24h Chl-a CRPS={chla.get('crps')} ΔCRPS={chla.get('delta_crps_vs_persistence')} cov90={chla.get('coverage_90')} scale={interval_scale:.3f}")
if __name__ == '__main__':
    main()
