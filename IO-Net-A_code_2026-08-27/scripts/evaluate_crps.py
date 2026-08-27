from __future__ import annotations
import argparse
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.baselines import QUANTILE_BASELINE_TYPES, TORCH_BASELINE_TYPES
from scripts.experiment_utils import load_artifact, load_scalers, setup_no_train_run
from scripts.npz_dataset import NPZDataset, load_meta, split_npz_paths
from scripts.probabilistic_eval import build_e2_probabilistic_metrics, persistence_pred_obs, predict_point_forecast, predict_quantile_baseline, save_probabilistic_metrics

def pick_device(name: str) -> torch.device:
    if name == 'cpu':
        return torch.device('cpu')
    if name in ('cuda', 'auto', 'gpu') and (not torch.cuda.is_available()):
        return torch.device('cpu')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['persistence', 'lstm', 'grud', 'qlstm'])
    ap.add_argument('--proc-dir', type=Path, required=True)
    ap.add_argument('--results-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--split', default='test')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--denormalize', type=int, default=1)
    ap.add_argument('--horizons', default='1,2,3,6')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--min-valid-n', type=int, default=30)
    args = ap.parse_args()
    meta = load_meta(args.proc_dir)
    device = pick_device(args.device)
    horizons = tuple((int(h) for h in args.horizons.split(',') if h.strip()))
    scalers = load_scalers(args.proc_dir / 'scalers.json') if args.denormalize else None
    paths = split_npz_paths(args.proc_dir)
    loader = DataLoader(NPZDataset(paths[args.split]), batch_size=args.batch_size, shuffle=False)
    persist_met = None
    quantile_levels = None
    if args.model == 'persistence':
        args.results_dir.mkdir(parents=True, exist_ok=True)
        if not (args.results_dir / 'run_config.json').exists():
            setup_no_train_run(args.model, args.proc_dir, args.results_dir, args.seed)
        pred, obs = persistence_pred_obs(args.proc_dir, args.split, args.batch_size)
    else:
        ckpt = args.results_dir / 'best.pt'
        if not ckpt.exists():
            raise SystemExit(f'Missing checkpoint: {ckpt}')
        artifact = load_artifact(args.model, args.results_dir, meta, device)
        if isinstance(artifact, QUANTILE_BASELINE_TYPES) or args.model == 'qlstm':
            pred, obs = predict_quantile_baseline(artifact, loader, device)
            quantile_levels = pred.get('quantile_levels')
            p_pred, _ = persistence_pred_obs(args.proc_dir, args.split, args.batch_size)
            persist_met = build_e2_probabilistic_metrics(p_pred, obs, meta, scalers, bool(args.denormalize), horizons, args.min_valid_n)
        else:
            is_baseline = isinstance(artifact, TORCH_BASELINE_TYPES)
            pred, obs = predict_point_forecast(artifact, loader, device, is_baseline)
    met = build_e2_probabilistic_metrics(pred, obs, meta, scalers, bool(args.denormalize), horizons, args.min_valid_n, quantile_levels, persist_met)
    out = args.out or args.results_dir / 'E2_crps.json'
    save_probabilistic_metrics(out, args.model, met)
    print(f'[evaluate_crps] saved -> {out}')
    chla = met.get('by_target', {}).get('chla', {}).get('h6', {})
    print(f"[evaluate_crps] 24h Chl-a CRPS={chla.get('crps')} ΔCRPS={chla.get('delta_crps_vs_persistence')} cov90={chla.get('coverage_90')}")
if __name__ == '__main__':
    main()
