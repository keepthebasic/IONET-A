from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.utils.data import DataLoader
from scripts.experiment_utils import denormalize_tensor, load_scalers, persistence_predict
from scripts.metrics import compute_masked_metrics
from scripts.metrics_probabilistic import coverage, crps_deterministic, crps_from_quantiles, delta_crps, interval_width
from scripts.npz_dataset import NPZDataset, load_meta
DEFAULT_QUANTILES = (0.1, 0.5, 0.9)

def load_split_arrays(proc_dir: Path, split: str) -> dict[str, np.ndarray]:
    data = np.load(proc_dir / f'{split}.npz', allow_pickle=False)
    return {k: data[k] for k in data.files}

@torch.no_grad()
def predict_quantiles(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    fc_q, fc_med, soft_list = ([], [], [])
    soft_y, soft_m, fc_y, fc_m = ([], [], [], [])
    q_levels: np.ndarray | None = None
    for batch in loader:
        x = batch['x'].to(device)
        mask = batch['mask'].to(device)
        exo = batch.get('exo')
        if exo is not None:
            exo = exo.to(device)
        if getattr(model, 'uses_soft_context', False):
            out = model(x, mask, exo, soft_y=batch['soft_y'].to(device), soft_y_mask=batch['soft_y_mask'].to(device), soft_lab_mask=batch['soft_lab_mask'].to(device) if 'soft_lab_mask' in batch else None)
        else:
            out = model(x, mask, exo)
        fq = out['forecast_quantiles'].cpu().numpy()
        fc_q.append(fq)
        fc_med.append(out['forecast'].cpu().numpy())
        soft_list.append(out['soft'].cpu().numpy())
        if q_levels is None:
            q_levels = out['quantile_levels'].cpu().numpy()
        soft_y.append(batch['soft_y'].numpy())
        soft_m.append(batch['soft_y_mask'].numpy())
        fc_y.append(batch['forecast_y'].numpy())
        fc_m.append(batch['forecast_y_mask'].numpy())
    return ({'forecast_quantiles': np.concatenate(fc_q, 0), 'forecast': np.concatenate(fc_med, 0), 'soft': np.concatenate(soft_list, 0), 'quantile_levels': q_levels if q_levels is not None else np.array(DEFAULT_QUANTILES)}, {'soft': np.concatenate(soft_y, 0), 'soft_mask': np.concatenate(soft_m, 0), 'forecast': np.concatenate(fc_y, 0), 'forecast_mask': np.concatenate(fc_m, 0)})

@torch.no_grad()
def predict_point_forecast(model: torch.nn.Module, loader: DataLoader, device: torch.device, is_baseline: bool=False) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    soft_list, fc_list = ([], [])
    soft_y, soft_m, fc_y, fc_m = ([], [], [], [])
    for batch in loader:
        x = batch['x'].to(device)
        mask = batch['mask'].to(device)
        if is_baseline:
            out = model(x, mask)
        else:
            exo = batch.get('exo')
            if exo is not None:
                exo = exo.to(device)
            out = model(x, mask, exo)
        soft_list.append(out['soft'].cpu().numpy())
        fc_list.append(out['forecast'].cpu().numpy())
        soft_y.append(batch['soft_y'].numpy())
        soft_m.append(batch['soft_y_mask'].numpy())
        fc_y.append(batch['forecast_y'].numpy())
        fc_m.append(batch['forecast_y_mask'].numpy())
    pred = {'soft': np.concatenate(soft_list, 0), 'forecast': np.concatenate(fc_list, 0)}
    obs = {'soft': np.concatenate(soft_y, 0), 'soft_mask': np.concatenate(soft_m, 0), 'forecast': np.concatenate(fc_y, 0), 'forecast_mask': np.concatenate(fc_m, 0)}
    return (pred, obs)

@torch.no_grad()
def predict_quantile_baseline(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    fc_q, fc_med = ([], [])
    fc_y, fc_m = ([], [])
    q_levels: np.ndarray | None = None
    for batch in loader:
        out = model(batch['x'].to(device), batch['mask'].to(device))
        fq = out['forecast_quantiles'].cpu().numpy()
        fc_q.append(fq)
        fc_med.append(out['forecast'].cpu().numpy())
        if q_levels is None:
            q_levels = out['quantile_levels'].detach().cpu().numpy()
        fc_y.append(batch['forecast_y'].numpy())
        fc_m.append(batch['forecast_y_mask'].numpy())
    pred = {'forecast_quantiles': np.concatenate(fc_q, 0), 'forecast': np.concatenate(fc_med, 0), 'quantile_levels': q_levels if q_levels is not None else np.array(DEFAULT_QUANTILES)}
    obs = {'forecast': np.concatenate(fc_y, 0), 'forecast_mask': np.concatenate(fc_m, 0)}
    return (pred, obs)

def persistence_pred_obs(proc_dir: Path, split: str='test', batch_size: int=64) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    meta = load_meta(proc_dir)
    paths = proc_dir / f'{split}.npz'
    train_data = load_split_arrays(proc_dir, 'train')
    test_data = load_split_arrays(proc_dir, split)
    loader = DataLoader(NPZDataset(paths), batch_size=batch_size, shuffle=False)
    xs, ms, sy, sm, fy, fm = ([], [], [], [], [], [])
    for batch in loader:
        xs.append(batch['x'].numpy())
        ms.append(batch['mask'].numpy())
        sy.append(batch['soft_y'].numpy())
        sm.append(batch['soft_y_mask'].numpy())
        fy.append(batch['forecast_y'].numpy())
        fm.append(batch['forecast_y_mask'].numpy())
    x = np.concatenate(xs, 0)
    mask = np.concatenate(ms, 0)
    pred = persistence_predict(x, mask, np.concatenate(sy, 0), np.concatenate(sm, 0), train_data['soft_y'], train_data['soft_y_mask'], n_forecast=len(meta['forecast_vars']), pred_len=int(meta['pred_len']), soft_vars=list(meta['soft_vars']))
    obs = {'soft': np.concatenate(sy, 0), 'soft_mask': np.concatenate(sm, 0), 'forecast': np.concatenate(fy, 0), 'forecast_mask': np.concatenate(fm, 0)}
    return (pred, obs)

def _maybe_denorm(arr: np.ndarray, var: str, scalers: dict | None, denormalize_flag: bool) -> np.ndarray:
    if not denormalize_flag or not scalers:
        return arr
    return denormalize_tensor(arr, [var], scalers).reshape(arr.shape)

def evaluate_forecast_horizon(pred: np.ndarray, obs: np.ndarray, mask: np.ndarray, var: str, scalers: dict | None, denormalize_flag: bool, min_valid_n: int, quantile_preds: np.ndarray | None=None, quantile_levels: np.ndarray | None=None) -> dict[str, Any]:
    p = _maybe_denorm(pred, var, scalers, denormalize_flag)
    o = _maybe_denorm(obs, var, scalers, denormalize_flag)
    m = mask
    n_valid = int((m > 0).sum()) if m is not None else int(np.isfinite(o).sum())
    row: dict[str, Any] = {'n_valid': n_valid}
    if n_valid < min_valid_n:
        row.update({'crps': float('nan'), 'nse': float('nan'), 'rmse': float('nan'), 'coverage_90': float('nan'), 'interval_width_90': float('nan'), 'nse_suppressed': True})
        return row
    row['crps'] = crps_deterministic(p, o, m)
    nse_row = compute_masked_metrics(p, o, m, metrics=('nse', 'rmse'))
    row['nse'] = nse_row['nse']
    row['rmse'] = nse_row['rmse']
    if quantile_preds is not None and quantile_levels is not None:
        q_denorm = np.stack([_maybe_denorm(quantile_preds[i], var, scalers, denormalize_flag) for i in range(quantile_preds.shape[0])], axis=0)
        row['crps'] = crps_from_quantiles(quantile_levels, q_denorm, o, m)
        q_lo = q_denorm[0]
        q_hi = q_denorm[-1]
        row['coverage_90'] = coverage(q_lo, q_hi, o, m)
        row['interval_width_90'] = interval_width(q_lo, q_hi, m)
    return row

def build_e2_probabilistic_metrics(pred: dict[str, np.ndarray], obs: dict[str, np.ndarray], meta: dict[str, Any], scalers: dict | None, denormalize_flag: bool=True, horizons: tuple[int, ...]=(1, 2, 3, 6), min_valid_n: int=30, quantile_levels: np.ndarray | None=None, persistence_metrics: dict[str, Any] | None=None) -> dict[str, Any]:
    met: dict[str, Any] = {'by_target': {}, 'by_horizon': {}}
    fq = pred.get('forecast_quantiles')
    q_levels = quantile_levels
    if fq is not None and q_levels is None:
        q_levels = np.array(DEFAULT_QUANTILES)
    for var in meta['forecast_vars']:
        i = meta['forecast_vars'].index(var)
        met['by_target'][var] = {}
        for h in horizons:
            hi = h - 1
            key = f'h{h}'
            q_slice = None
            if fq is not None:
                q_slice = fq[:, :, i, hi].T
            row = evaluate_forecast_horizon(pred['forecast'][:, i, hi], obs['forecast'][:, i, hi], obs['forecast_mask'][:, i, hi], var, scalers, denormalize_flag, min_valid_n, q_slice, q_levels)
            if persistence_metrics is not None:
                p_row = persistence_metrics.get('by_target', {}).get(var, {}).get(key, {})
                p_crps = p_row.get('crps')
                if p_crps is not None and np.isfinite(p_crps) and np.isfinite(row.get('crps', float('nan'))):
                    row['delta_crps_vs_persistence'] = delta_crps(row['crps'], p_crps)
            met['by_target'][var][key] = row
            met['by_horizon'].setdefault(key, {})[var] = row
    return met

def save_probabilistic_metrics(path: Path, model: str, metrics: dict[str, Any], extra: dict[str, Any] | None=None) -> None:
    payload = {'exp': 'E2', 'model': model, 'metrics': metrics}
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(__import__('json').dumps(payload, indent=2), encoding='utf-8')
