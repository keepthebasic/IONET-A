from __future__ import annotations
import argparse
import json
import shutil
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.experiment_utils import NO_TRAIN_MODELS, SKLEARN_MODELS, parse_seeds, run_subprocess

def apply_random_mask(npz_path: Path, out_path: Path, rate: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    data = dict(np.load(npz_path, allow_pickle=False))
    mask = data['mask'].copy()
    n, c, l = mask.shape
    for i in range(n):
        for ch in range(c):
            valid = np.where(mask[i, ch] > 0)[0]
            if len(valid) == 0:
                continue
            k = int(len(valid) * rate)
            if k <= 0:
                continue
            drop = rng.choice(valid, size=k, replace=False)
            mask[i, ch, drop] = 0
            data['x'][i, ch, drop] = 0.0
    data['mask'] = mask
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **data)

def _linear_fill_1d(x: np.ndarray, m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out = np.asarray(x, dtype=np.float64).copy()
    mout = np.asarray(m, dtype=np.float64).copy()
    valid = np.where(mout > 0)[0]
    if len(valid) == 0:
        return (out, mout)
    observed_vals = out[valid].copy()
    if len(valid) == 1:
        out[:] = out[valid[0]]
        mout[:] = 1.0
    else:
        t = np.arange(len(out), dtype=np.float64)
        out = np.interp(t, valid.astype(np.float64), out[valid])
        mout[:] = 1.0
    if not np.allclose(out[valid], observed_vals, rtol=0.0, atol=0.0):
        raise RuntimeError('linear impute altered an observed value (implementation bug)')
    return (out, mout)

def apply_linear_impute(npz_path: Path, out_path: Path) -> None:
    data = dict(np.load(npz_path, allow_pickle=False))
    x_in = np.asarray(data['x'], dtype=np.float64)
    mask_in = np.asarray(data['mask'], dtype=np.float64)
    x = x_in.copy()
    mask = mask_in.copy()
    n, c, _l = mask.shape
    for i in range(n):
        for ch in range(c):
            x[i, ch], mask[i, ch] = _linear_fill_1d(x[i, ch], mask[i, ch])
    observed = mask_in > 0
    if observed.any() and (not np.allclose(x[observed], x_in[observed], rtol=0.0, atol=0.0)):
        raise RuntimeError(f'apply_linear_impute changed observed entries in {npz_path}')
    data['x'] = x.astype(np.float32)
    data['mask'] = mask.astype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **data)

def _locf_fill_1d(x: np.ndarray, m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out = x.astype(np.float64, copy=True)
    mout = np.ones_like(m, dtype=np.float64)
    valid = m > 0
    if not valid.any():
        out[:] = 0.0
        return (out, mout)
    last = None
    for t in range(out.shape[0]):
        if valid[t]:
            last = float(out[t])
        elif last is not None:
            out[t] = last
    first = float(out[valid][0])
    for t in range(out.shape[0]):
        if valid[t]:
            break
        out[t] = first
    return (out, mout)

def apply_locf_impute(npz_path: Path, out_path: Path) -> None:
    data = dict(np.load(npz_path, allow_pickle=False))
    x_in = np.asarray(data['x'], dtype=np.float64)
    mask_in = np.asarray(data['mask'], dtype=np.float64)
    x = x_in.copy()
    mask = mask_in.copy()
    n, c, _l = mask.shape
    for i in range(n):
        for ch in range(c):
            x[i, ch], mask[i, ch] = _locf_fill_1d(x[i, ch], mask[i, ch])
    observed = mask_in > 0
    if observed.any() and (not np.allclose(x[observed], x_in[observed], rtol=0.0, atol=0.0)):
        raise RuntimeError(f'apply_locf_impute changed observed entries in {npz_path}')
    data['x'] = x.astype(np.float32)
    data['mask'] = mask.astype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **data)

def _copy_sidecars(src_proc: Path, dst_proc: Path) -> None:
    for name in ('meta.json', 'scalers.json'):
        src = src_proc / name
        if src.exists():
            shutil.copy2(src, dst_proc / name)

def _run_complete(out: Path, model: str) -> bool:
    if not (out / 'E2_metrics.json').exists():
        return False
    if model in NO_TRAIN_MODELS:
        return (out / 'run_config.json').exists()
    if model in SKLEARN_MODELS:
        return (out / 'sklearn_model.pkl').exists()
    return (out / 'best.pt').exists()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='E5a')
    parser.add_argument('--mode', default='native_missing')
    parser.add_argument('--proc-dir', type=Path, required=True)
    parser.add_argument('--baselines', default='persistence,pls,xgboost,lstm,other models')
    parser.add_argument('--model', default=None)
    parser.add_argument('--tasks', default='soft,forecast')
    parser.add_argument('--seeds', default='0,1,2,3,4')
    parser.add_argument('--mask-rate', type=float, default=0.3)
    parser.add_argument('--imputer', default='linear')
    parser.add_argument('--hyperparams', type=Path, default=None)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu', 'auto'])
    args = parser.parse_args()
    seeds = parse_seeds(args.seeds)
    models = [args.model] if args.model else [m.strip() for m in args.baselines.split(',') if m.strip()]
    orig_proc = Path(args.proc_dir)
    proc_dir = orig_proc
    work_dir = args.out_dir / 'data'
    work_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == 'random_mask':
        for split in ('train', 'val', 'test'):
            src = orig_proc / f'{split}.npz'
            if src.exists():
                apply_random_mask(src, work_dir / f'{split}.npz', args.mask_rate, 42)
        _copy_sidecars(orig_proc, work_dir)
        if not (work_dir / 'meta.json').exists():
            raise SystemExit(f'random_mask needs meta.json from {orig_proc}')
        proc_dir = work_dir
    elif args.mode in ('impute_then_ml', 'random_mask_impute'):
        if args.imputer != 'linear':
            raise SystemExit(f"Unsupported --imputer={args.imputer!r}; only 'linear' is implemented.")
        for split in ('train', 'val', 'test'):
            src = orig_proc / f'{split}.npz'
            if not src.exists():
                continue
            mid = work_dir / f'{split}_masked.npz' if args.mode == 'random_mask_impute' else src
            if args.mode == 'random_mask_impute':
                apply_random_mask(src, mid, args.mask_rate, 42)
            apply_linear_impute(mid if args.mode == 'random_mask_impute' else src, work_dir / f'{split}.npz')
            if args.mode == 'random_mask_impute' and mid.exists() and (mid != work_dir / f'{split}.npz'):
                mid.unlink(missing_ok=True)
        _copy_sidecars(orig_proc, work_dir)
        if not (work_dir / 'meta.json').exists():
            raise SystemExit(f'{args.mode} needs meta.json from {orig_proc}')
        proc_dir = work_dir
        if args.model is None and args.baselines == 'persistence,pls,xgboost,lstm,other models':
            models = ['lstm']
    results = []
    for model in models:
        for seed in seeds:
            out = args.out_dir / f'{model}_seed{seed}'
            if _run_complete(out, model):
                print(f'[run_robustness] skip existing {model} seed={seed} -> {out}')
                results.append(json.loads((out / 'E2_metrics.json').read_text(encoding='utf-8')))
                continue
            cmd = [sys.executable, str(ROOT / 'scripts' / 'train.py'), '--model', model, '--proc-dir', str(proc_dir), '--tasks', args.tasks, '--seed', str(seed), '--out-dir', str(out), '--device', args.device]
            if args.hyperparams:
                cmd += ['--hyperparams', str(args.hyperparams)]
            if model == 'persistence':
                cmd += ['--epochs', '1']
            run_subprocess(cmd)
            ev = [sys.executable, str(ROOT / 'scripts' / 'evaluate.py'), '--exp', 'E2', '--model', model, '--results-dir', str(out), '--proc-dir', str(proc_dir), '--denormalize', '1', '--targets', 'do,chla', '--device', args.device, '--scalers', str(orig_proc / 'scalers.json'), '--out', str(out / 'E2_metrics.json')]
            run_subprocess(ev)
            if (out / 'E2_metrics.json').exists():
                results.append(json.loads((out / 'E2_metrics.json').read_text(encoding='utf-8')))
    summary = args.out_dir / 'summary.json'
    summary.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(f'[run_robustness] saved -> {summary}')
if __name__ == '__main__':
    main()
