from __future__ import annotations
import numpy as np

def _valid_triplets(pred: np.ndarray, obs: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    obs = np.asarray(obs, dtype=np.float64).reshape(-1)
    if mask is None:
        valid = np.isfinite(pred) & np.isfinite(obs)
    else:
        m = np.asarray(mask, dtype=np.float64).reshape(-1)
        valid = (m > 0) & np.isfinite(pred) & np.isfinite(obs)
    return (pred[valid], obs[valid])

def pinball(pred_q: np.ndarray, obs: np.ndarray, quantile: float, mask: np.ndarray | None=None) -> float:
    p, o = _valid_triplets(pred_q, obs, mask)
    if len(o) == 0:
        return float('nan')
    q = float(quantile)
    err = o - p
    return float(np.mean(np.maximum(q * err, (q - 1.0) * err)))

def crps_from_quantiles(quantile_levels: np.ndarray | list[float], quantile_preds: np.ndarray, obs: np.ndarray, mask: np.ndarray | None=None) -> float:
    levels = np.asarray(quantile_levels, dtype=np.float64)
    preds = np.asarray(quantile_preds, dtype=np.float64)
    if preds.ndim == 1:
        return pinball(preds, obs, float(levels[0]) if levels.size else 0.5, mask)
    if preds.shape[0] != levels.size:
        raise ValueError(f'quantile_preds {preds.shape} vs levels {levels.shape}')
    o = np.asarray(obs, dtype=np.float64).reshape(-1)
    m = None if mask is None else np.asarray(mask, dtype=np.float64).reshape(-1)
    if m is not None:
        valid = (m > 0) & np.isfinite(o)
        for row in preds:
            valid &= np.isfinite(row.reshape(-1))
        if not np.any(valid):
            return float('nan')
        o = o[valid]
        preds = preds[:, valid]
    else:
        valid = np.isfinite(o)
        for row in preds:
            valid &= np.isfinite(row.reshape(-1))
        if not np.any(valid):
            return float('nan')
        o = o[valid]
        preds = preds[:, valid]
    order = np.argsort(levels)
    levels = levels[order]
    preds = preds[order]
    pinballs = []
    for lv, row in zip(levels, preds):
        err = o - row
        pinballs.append(np.maximum(lv * err, (lv - 1.0) * err))
    pinballs = np.stack(pinballs, axis=0)
    if levels.size == 1:
        return float(pinballs.mean())
    crps = 0.0
    for i in range(len(levels) - 1):
        dq = levels[i + 1] - levels[i]
        avg_pb = 0.5 * (pinballs[i] + pinballs[i + 1])
        crps += float(np.mean(avg_pb) * dq)
    return 2.0 * crps

def crps_deterministic(pred: np.ndarray, obs: np.ndarray, mask: np.ndarray | None=None) -> float:
    p, o = _valid_triplets(pred, obs, mask)
    if len(o) == 0:
        return float('nan')
    return float(np.mean(np.abs(p - o)))

def coverage(lower: np.ndarray, upper: np.ndarray, obs: np.ndarray, mask: np.ndarray | None=None) -> float:
    p_lo, o = _valid_triplets(lower, obs, mask)
    p_hi, _ = _valid_triplets(upper, obs, mask)
    if len(o) == 0:
        return float('nan')
    inside = (o >= p_lo) & (o <= p_hi)
    return float(np.mean(inside))

def interval_width(lower: np.ndarray, upper: np.ndarray, mask: np.ndarray | None=None) -> float:
    lo = np.asarray(lower, dtype=np.float64).reshape(-1)
    hi = np.asarray(upper, dtype=np.float64).reshape(-1)
    if mask is not None:
        m = np.asarray(mask, dtype=np.float64).reshape(-1)
        valid = (m > 0) & np.isfinite(lo) & np.isfinite(hi)
        lo, hi = (lo[valid], hi[valid])
    if lo.size == 0:
        return float('nan')
    return float(np.mean(hi - lo))

def delta_crps(model_crps: float, baseline_crps: float) -> float:
    if not np.isfinite(model_crps) or not np.isfinite(baseline_crps):
        return float('nan')
    return float(baseline_crps - model_crps)
