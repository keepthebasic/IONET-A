from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from models.ionet_a import build_ionet_a
from scripts.experiment_utils import load_scalers
from scripts.backbone import resolve_backbone
from scripts.npz_dataset import NPZDataset, load_meta, split_npz_paths
from scripts.probabilistic_eval import build_e2_probabilistic_metrics, predict_quantiles, save_probabilistic_metrics

class ProbeLOODataset(torch.utils.data.Dataset):

    def __init__(self, base: NPZDataset, channel_idx: int, mode: str='mask'):
        self.base = base
        self.channel_idx = channel_idx
        self.mode = mode

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        batch = self.base[idx]
        batch = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}
        batch['mask'][self.channel_idx, :] = 0.0
        if self.mode == 'zero':
            batch['x'][self.channel_idx, :] = 0.0
        return batch

def pick_device(name: str) -> torch.device:
    if name == 'cpu':
        return torch.device('cpu')
    return torch.device('cuda' if torch.cuda.is_available() and name != 'cpu' else 'cpu')

def _load_model(backbone: str, ckpt_path: Path, meta: dict, device: torch.device):
    if backbone not in ('ionet_a', 'ionet_lite'):
        raise SystemExit(f'Unsupported backbone {backbone!r}; this release ships IO-Net-A only')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get('ionet_a_cfg') or ckpt.get('ionet_lite_cfg') or ckpt.get('model_cfg') or {}
    kw = dict(in_channels=len(meta['easy_vars']), seq_len=int(meta['seq_len']), pred_len=int(meta['pred_len']), n_forecast=len(meta['forecast_vars']))
    for k in ('hidden', 'dropout', 'quantiles', 'use_persist_anchor', 'use_diurnal_template', 'use_mask_gate', 'use_lab_age', 'use_monotonic_quantiles', 'width_reg_weight'):
        if k in cfg:
            kw[k] = cfg[k]
    model = build_ionet_a(**kw)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def eval_probe_loo(ckpt_dir: Path, proc_dir: Path, channel: str, backbone: str, device: str, batch_size: int=64, mode: str='mask') -> Path:
    meta = load_meta(proc_dir)
    if channel not in meta['easy_vars']:
        raise SystemExit(f"Unknown channel {channel}; have {meta['easy_vars']}")
    ch_idx = meta['easy_vars'].index(channel)
    dev = pick_device(device)
    model = _load_model(backbone, ckpt_dir / 'best.pt', meta, dev)
    test_path = split_npz_paths(proc_dir)['test']
    loader = DataLoader(ProbeLOODataset(NPZDataset(test_path), ch_idx, mode=mode), batch_size=batch_size, shuffle=False)
    pred, obs = predict_quantiles(model, loader, dev)
    scalers = load_scalers(proc_dir / 'scalers.json')
    met = build_e2_probabilistic_metrics(pred, obs, meta, scalers, True, (1, 2, 3, 6), 30, pred['quantile_levels'])
    tag = f'{channel}_{mode}'
    out = ckpt_dir / f'E2_crps_loo_{tag}.json'
    save_probabilistic_metrics(out, backbone, met, extra={'probe_loo': channel, 'drop_mode': mode})
    return out

def run(cmd: list[str]) -> None:
    print('+', ' '.join(cmd), flush=True)
    env = os.environ.copy()
    env.setdefault('MKL_THREADING_LAYER', 'GNU')
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
    if proc.returncode != 0:
        raise SystemExit(f"failed ({proc.returncode}): {' '.join(cmd)}")

def _metric(path: Path) -> float | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding='utf-8'))
    chla = d.get('metrics', {}).get('by_target', {}).get('chla', {}).get('h6', {})
    v = chla.get('crps')
    return float(v) if v is not None else None

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--proc-dir', type=Path, default=ROOT / 'data/processed/lakebed_BVR_4h')
    ap.add_argument('--out-dir', type=Path, default=ROOT / 'results/BVR_4h/phaseB/probe_loo')
    ap.add_argument('--channels', default='temp,turbidity,do,ec')
    ap.add_argument('--seeds', default='0,1,2,3,4')
    ap.add_argument('--backbone', default='ionet_a', choices=['ionet_a'])
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--drop-mode', default='mask', choices=['mask', 'zero'], help='mask=availability dropout (recommended); zero=legacy x:=0+mask:=0')
    args = ap.parse_args()
    spec = resolve_backbone(args.backbone)
    train_script = spec.train_script
    eval_script = spec.eval_script
    hp = spec.hp_file
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    channels = [c.strip() for c in args.channels.split(',') if c.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, list[float]]] = {c: {'crps': [], 'delta_crps': []} for c in channels}
    for seed in seeds:
        ckpt_dir = args.out_dir / f'seed{seed}'
        if not (ckpt_dir / 'best.pt').exists() or args.force:
            run([PY, str(ROOT / f'scripts/{train_script}'), '--proc-dir', str(args.proc_dir), '--hyperparams', str(ROOT / f'configs/{hp}'), '--seed', str(seed), '--device', args.device, '--out-dir', str(ckpt_dir)] + (['--force'] if args.force else []))
        native_crps = ckpt_dir / 'E2_crps_native.json'
        if not native_crps.exists() or args.force:
            run([PY, str(ROOT / f'scripts/{eval_script}'), '--proc-dir', str(args.proc_dir), '--results-dir', str(ckpt_dir), '--out', str(native_crps), '--denormalize', '1', '--calibrate', '0', '--device', args.device])
        for ch in channels:
            out = eval_probe_loo(ckpt_dir, args.proc_dir, ch, args.backbone, args.device, mode=args.drop_mode)
            crps = _metric(out)
            native = _metric(native_crps)
            delta = crps - native if crps is not None and native is not None else None
            print(f'[probe_loo] seed={seed} drop={ch} mode={args.drop_mode} CRPS={crps} Δvs_native={delta}', flush=True)
            summary[ch]['crps'].append(crps if crps is not None else float('nan'))
            if delta is not None:
                summary[ch]['delta_crps'].append(delta)
    print('\n=== Probe LOO summary (24h Chl-a CRPS) ===', flush=True)
    for ch, d in summary.items():
        vals = [v for v in d.get('crps', []) if np.isfinite(v)]
        if vals:
            mu = sum(vals) / len(vals)
            print(f'  drop {ch}: CRPS={mu:.4f}  n={len(vals)}', flush=True)
    (args.out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')
if __name__ == '__main__':
    main()
