from __future__ import annotations
import json
import os
import pickle
import random
import subprocess
import sys
from pathlib import Path
from typing import Any
os.environ.setdefault('MKL_THREADING_LAYER', 'GNU')
os.environ.setdefault('OMP_NUM_THREADS', os.environ.get('OMP_NUM_THREADS', '4'))
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.baselines import BaselineGRUD, BaselineInformer, BaselineLSTM, BaselineMTLSTM, BaselinePatchTST, BaselineQuantileLSTM, BaselineTCN, SklearnBundle, TORCH_BASELINE_TYPES, predict_sklearn_baseline
from scripts.common import EASY_VARS, SOFT_VARS
from scripts.metrics import compute_masked_metrics
from scripts.npz_dataset import NPZDataset, load_meta, split_npz_paths
TORCH_MODELS = {'lstm', 'qlstm', 'tcn', 'mtlstm', 'informer', 'patchtst', 'grud'}
SKLEARN_MODELS = {'pls', 'xgboost'}
NO_TRAIN_MODELS = {'persistence'}

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_seeds(text: str) -> list[int]:
    return [int(s.strip()) for s in text.split(',') if s.strip()]

def load_scalers(path: Path) -> dict[str, dict[str, float]]:
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def denormalize(values: np.ndarray, var: str, scalers: dict[str, dict[str, float]]) -> np.ndarray:
    if var not in scalers:
        return values
    m, s = (scalers[var]['mean'], scalers[var]['std'])
    return values.astype(np.float64) * s + m

def denormalize_tensor(arr: np.ndarray, var_names: list[str], scalers: dict[str, dict[str, float]]) -> np.ndarray:
    out = arr.copy()
    if out.ndim == 1:
        return denormalize(out, var_names[0], scalers)
    for i, v in enumerate(var_names):
        if out.ndim == 2:
            out[:, i] = denormalize(out[:, i], v, scalers)
        elif out.ndim == 3:
            out[:, i, :] = denormalize(out[:, i, :], v, scalers)
    return out

def load_hyperparams(path: Path | None, model: str) -> dict[str, Any]:
    if path is None or not Path(path).exists():
        return {}
    with open(path, encoding='utf-8') as f:
        cfg = json.load(f)
    return dict(cfg.get(model, {}))

def apply_hyperparams_to_train_args(args: Any, hp: dict[str, Any]) -> None:
    if not hp:
        return
    mapping = {'lr': ('lr', float), 'batch_size': ('batch_size', int), 'd_model': ('d_model', int), 'hidden': ('hidden', int), 'num_layers': ('num_layers', int), 'dropout': ('dropout', float), 'n_heads': ('n_heads', int), 'patch_len': ('patch_len', int), 'stride': ('stride', int), 'factor': ('factor', int), 'max_depth': ('max_depth', int), 'learning_rate': ('xgb_lr', float), 'n_estimators': ('n_estimators', int), 'n_components': ('n_components', int), 'chla_pinball_weight': ('chla_pinball_weight', float)}
    for key, (attr, cast) in mapping.items():
        if key in hp:
            setattr(args, attr, cast(hp[key]))
    lw = hp.get('loss_weights')
    if isinstance(lw, dict):
        args.loss_weights = ','.join((f'{k}={v}' for k, v in lw.items()))
    elif isinstance(lw, str):
        args.loss_weights = lw

def setup_no_train_run(model: str, proc_dir: Path, out_dir: Path, seed: int, hyperparams: Path | None=None, tasks: str='soft,forecast') -> None:
    if model not in NO_TRAIN_MODELS:
        raise ValueError(f'setup_no_train_run only for {NO_TRAIN_MODELS}, got {model}')
    out_dir.mkdir(parents=True, exist_ok=True)
    hp = load_hyperparams(hyperparams, model)
    meta = load_meta(proc_dir)
    with open(out_dir / 'run_config.json', 'w', encoding='utf-8') as f:
        json.dump({'exp': 'joint', 'model': model, 'proc_dir': str(proc_dir), 'tasks': tasks, 'seed': seed, 'hyperparams': hp, 'soft_vars': list(meta.get('soft_vars', SOFT_VARS)), 'forecast_vars': list(meta.get('forecast_vars', ())), 'note': 'no training; evaluate-only baseline'}, f, indent=2, default=str)
    print(f"[train] {model}: evaluate-only -> {out_dir / 'run_config.json'}")

def require_npz_splits(proc_dir: Path) -> None:
    missing = [name for name, p in split_npz_paths(proc_dir).items() if not p.exists()]
    if missing:
        csv_guess = list(Path(proc_dir).glob('*_4h.csv'))
        hint = f'python scripts/data_process.py --input {csv_guess[0]} --out-dir {proc_dir}' if csv_guess else f'upload train.npz val.npz test.npz to {proc_dir}'
        raise FileNotFoundError(f"Missing NPZ splits in {proc_dir}: {', '.join(missing)}. Rebuild with: {hint}")

def persistence_predict(x: np.ndarray, mask: np.ndarray, soft_y: np.ndarray, soft_y_mask: np.ndarray, train_soft_y: np.ndarray | None, train_soft_mask: np.ndarray | None, n_forecast: int, pred_len: int, soft_vars: list[str] | None=None) -> dict[str, np.ndarray]:
    n = len(x)
    soft_vars = soft_vars or list(SOFT_VARS)
    n_soft = len(soft_vars)
    chla_idx = soft_vars.index('chla') if 'chla' in soft_vars else min(2, n_soft - 1)
    do_x_idx = list(EASY_VARS).index('do')
    soft = np.zeros((n, n_soft), dtype=np.float32)
    train_means: dict[int, float] = {}
    if train_soft_y is not None and train_soft_mask is not None:
        for i in range(n_soft):
            valid = train_soft_mask[:, i] > 0
            train_means[i] = float(train_soft_y[valid, i].mean()) if valid.any() else 0.0
    for t in range(n):
        for i in range(n_soft):
            hist = soft_y[:t, i]
            hist_m = soft_y_mask[:t, i] > 0
            if hist_m.any():
                soft[t, i] = hist[hist_m][-1]
            else:
                soft[t, i] = train_means.get(i, 0.0)
    forecast = np.zeros((n, n_forecast, pred_len), dtype=np.float32)
    do_last = x[:, do_x_idx, -1]
    chla_t0 = soft[:, chla_idx]
    for h in range(pred_len):
        forecast[:, 0, h] = do_last
        if n_forecast > 1:
            forecast[:, 1, h] = chla_t0
    return {'soft': soft, 'forecast': forecast}

@torch.no_grad()
def predict_torch(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    soft_list, fc_list = ([], [])
    for batch in loader:
        x = batch['x'].to(device)
        mask = batch['mask'].to(device)
        exo = batch.get('exo')
        if exo is not None:
            exo = exo.to(device)
        if isinstance(model, TORCH_BASELINE_TYPES):
            out = model(x, mask)
        else:
            out = model(x, mask, exo)
        soft_list.append(out['soft'].cpu().numpy())
        fc_list.append(out['forecast'].cpu().numpy())
    return {'soft': np.concatenate(soft_list, 0), 'forecast': np.concatenate(fc_list, 0)}

def build_torch_from_ckpt(ckpt: dict[str, Any], meta: dict[str, Any], model_name: str, device: torch.device) -> nn.Module:
    pred_len = int(meta['pred_len'])
    n_soft = len(meta['soft_vars'])
    n_forecast = len(meta['forecast_vars'])
    cin = len(meta['easy_vars'])
    saved = ckpt.get('model', model_name)
    seq_len = int(meta['seq_len'])
    if saved == 'lstm':
        model: nn.Module = BaselineLSTM(cin, n_soft=n_soft, n_forecast=n_forecast, pred_len=pred_len)
    elif saved == 'qlstm':
        args = ckpt.get('args') or {}
        model = BaselineQuantileLSTM(cin, hidden=int(args.get('hidden', 64)), n_soft=n_soft, n_forecast=n_forecast, pred_len=pred_len, num_layers=int(args.get('num_layers', 2)), dropout=float(args.get('dropout', 0.1)))
    elif saved == 'grud':
        args = ckpt.get('args') or {}
        model = BaselineGRUD(cin, hidden=int(args.get('hidden', 64)), n_soft=n_soft, n_forecast=n_forecast, pred_len=pred_len, dropout=float(args.get('dropout', 0.1)))
    elif saved == 'tcn':
        model = BaselineTCN(cin, n_soft=n_soft, n_forecast=n_forecast, pred_len=pred_len)
    elif saved == 'mtlstm':
        model = BaselineMTLSTM(cin, n_soft=n_soft, n_forecast=n_forecast, pred_len=pred_len)
    elif saved == 'informer':
        args = ckpt.get('args') or {}
        model = BaselineInformer(cin, d_model=int(args.get('d_model', 64)), n_soft=n_soft, n_forecast=n_forecast, pred_len=pred_len, n_heads=int(args.get('n_heads', 4)), num_layers=int(args.get('num_layers', 2)), dropout=float(args.get('dropout', 0.1)), factor=int(args.get('factor', 5)))
    elif saved == 'patchtst':
        args = ckpt.get('args') or {}
        model = BaselinePatchTST(cin, seq_len=seq_len, d_model=int(args.get('d_model', 64)), n_soft=n_soft, n_forecast=n_forecast, pred_len=pred_len, n_heads=int(args.get('n_heads', 4)), num_layers=int(args.get('num_layers', 2)), dropout=float(args.get('dropout', 0.1)), patch_len=int(args.get('patch_len', 16)), stride=int(args.get('stride', 8)))
    else:
        raise SystemExit(f'Unsupported checkpoint model: {saved}')
    missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)
    if missing or unexpected:
        print(f'[load] {saved}: missing={len(missing)} unexpected={len(unexpected)} (non-strict; check arch match if metrics look wrong)')
    bad = [k for k in missing if 'fuse.0.weight' in k or 'var_convs.0' in k]
    if bad:
        raise RuntimeError(f'Checkpoint architecture mismatch ({bad[:3]}).')
    model.to(device)
    model.eval()
    return model

def _torch_load_ckpt(path: Path, map_location: torch.device | str) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except Exception as first:
        import pathlib
        if os.name != 'nt':
            raise first
        local = getattr(pathlib, '_local', None)
        old_posix = pathlib.PosixPath
        old_local = getattr(local, 'PosixPath', None) if local is not None else None
        try:
            pathlib.PosixPath = pathlib.WindowsPath
            if local is not None:
                local.PosixPath = pathlib.WindowsPath
            return torch.load(path, map_location=map_location, weights_only=False)
        except Exception:
            raise first
        finally:
            pathlib.PosixPath = old_posix
            if local is not None and old_local is not None:
                local.PosixPath = old_local

def load_artifact(model_name: str, run_dir: Path, meta: dict[str, Any], device: torch.device):
    if model_name == 'persistence':
        return None
    pkl = run_dir / 'sklearn_model.pkl'
    if pkl.exists():
        with open(pkl, 'rb') as f:
            return pickle.load(f)
    ckpt = _torch_load_ckpt(run_dir / 'best.pt', map_location=device)
    return build_torch_from_ckpt(ckpt, meta, model_name, device)

def run_predictions(model_name: str, run_dir: Path | None, proc_dir: Path, split: str='test', batch_size: int=64, device: str='auto') -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    meta = load_meta(proc_dir)
    paths = split_npz_paths(proc_dir)
    ds = NPZDataset(paths[split])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    data = ds._store
    dev = torch.device('cuda' if torch.cuda.is_available() and device != 'cpu' else 'cpu')
    train_store = NPZDataset(paths['train'])._store
    if model_name == 'persistence':
        pred = persistence_predict(data['x'], data['mask'], data['soft_y'], data['soft_y_mask'], train_store['soft_y'], train_store['soft_y_mask'], len(meta['forecast_vars']), int(meta['pred_len']), meta['soft_vars'])
    else:
        if run_dir is None:
            raise ValueError(f'run_dir required for {model_name}')
        artifact = load_artifact(model_name, run_dir, meta, dev)
        if isinstance(artifact, SklearnBundle):
            pred = predict_sklearn_baseline(artifact, data['x'], data['mask'], len(meta['forecast_vars']), int(meta['pred_len']), n_soft=len(meta['soft_vars']))
        else:
            pred = predict_torch(artifact, loader, dev)
    obs = {'soft': data['soft_y'], 'soft_mask': data['soft_y_mask'], 'forecast': data['forecast_y'], 'forecast_mask': data['forecast_y_mask']}
    return (pred, obs, meta)

def metrics_with_denorm(pred: np.ndarray, obs: np.ndarray, mask: np.ndarray | None, var_names: list[str], scalers: dict | None, denormalize_flag: bool, metrics: tuple[str, ...], min_valid_n: int=30) -> dict[str, Any]:
    if denormalize_flag and scalers:
        pred = denormalize_tensor(pred, var_names, scalers)
        obs = denormalize_tensor(obs, var_names, scalers)
    result: dict[str, Any] = {}
    for i, var in enumerate(var_names):
        m = mask[:, i] if mask is not None and mask.ndim > 1 else mask
        p = pred[:, i] if pred.ndim > 1 else pred
        o = obs[:, i] if obs.ndim > 1 else obs
        n_valid = int((m > 0).sum()) if m is not None else int(np.isfinite(o).sum())
        row = compute_masked_metrics(p, o, m, metrics=metrics)
        row['n_valid'] = n_valid
        if 'nse' in row and n_valid < min_valid_n:
            row['nse'] = float('nan')
            row['nse_suppressed'] = True
        result[var] = row
    return result

def discover_run_dirs(results_root: Path, model: str, seeds: list[int]) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for seed in seeds:
        candidates = [results_root / f'{model}_seed{seed}', results_root / 'E1_E2' / f'{model}_seed{seed}']
        for p in results_root.rglob(f'{model}_seed{seed}'):
            candidates.append(p)
        for p in candidates:
            if p.exists() and ((p / 'best.pt').exists() or (p / 'sklearn_model.pkl').exists() or (p / 'run_config.json').exists()):
                out[seed] = p
                break
    return out

def run_subprocess(cmd: list[str], cwd: Path | None=None) -> None:
    print('[run]', ' '.join((str(c) for c in cmd)))
    env = os.environ.copy()
    env.setdefault('MKL_THREADING_LAYER', 'GNU')
    env.setdefault('OMP_NUM_THREADS', env.get('OMP_NUM_THREADS', '4'))
    env.setdefault('XGB_N_JOBS', env.get('XGB_N_JOBS', '4'))
    result = subprocess.run(cmd, cwd=cwd or ROOT, check=False, text=True, capture_output=True, env=env)
    if result.stdout:
        print(result.stdout, end='' if result.stdout.endswith('\n') else '\n')
    if result.stderr:
        print(result.stderr, end='' if result.stderr.endswith('\n') else '\n', file=sys.stderr)
    if result.returncode != 0:
        log_path = (cwd or ROOT) / 'results' / 'last_subprocess_failure.log'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(f"cmd: {' '.join((str(c) for c in cmd))}\n\n--- stdout ---\n{result.stdout}\n\n--- stderr ---\n{result.stderr}\n", encoding='utf-8')
        print(f'[run] details saved -> {log_path}', file=sys.stderr)
        print(f'[run] FAILED exit={result.returncode}', file=sys.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd, output=result.stdout, stderr=result.stderr)

def wilcoxon_pvalue(a: np.ndarray, b: np.ndarray) -> float:
    from scipy import stats
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 5:
        return float('nan')
    try:
        return float(stats.wilcoxon(a[mask], b[mask], alternative='two-sided').pvalue)
    except Exception:
        return float('nan')

def bootstrap_ci(values: np.ndarray, n_boot: int=1000, alpha: float=0.05, seed: int=42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    v = values[np.isfinite(values)]
    if len(v) == 0:
        return (float('nan'), float('nan'))
    boots = [float(np.mean(rng.choice(v, size=len(v), replace=True))) for _ in range(n_boot)]
    return (float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2)))
