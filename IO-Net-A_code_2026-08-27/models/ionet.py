from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
DEFAULT_QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)

def time_since_last_observation(mask: torch.Tensor) -> torch.Tensor:
    b, c, length = mask.shape
    deltas = mask.new_zeros(b, c, length)
    for t in range(1, length):
        deltas[:, :, t] = 1.0 + (1.0 - mask[:, :, t - 1]) * deltas[:, :, t - 1]
    return deltas

def normalize_delta(delta: torch.Tensor, seq_len: int) -> torch.Tensor:
    denom = max(float(seq_len - 1), 1.0)
    return torch.log1p(delta) / torch.log1p(torch.tensor(denom, device=delta.device, dtype=delta.dtype))

@dataclass
class IONetConfig:
    in_channels: int = 5
    seq_len: int = 168
    pred_len: int = 6
    n_forecast: int = 2
    hidden: int = 64
    dropout: float = 0.1
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES
    do_channel_index: int = 4
    chla_forecast_index: int = 1
    use_persist_anchor: bool = True

class _TCNBlock(nn.Module):

    def __init__(self, channels: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel, padding=pad, dilation=dilation)
        self.norm = nn.GroupNorm(1, channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = h[..., :x.size(-1)]
        h = self.drop(F.gelu(self.norm(h)))
        return x + h

class IONet(nn.Module):

    def __init__(self, cfg: IONetConfig | None=None, **kwargs):
        super().__init__()
        if cfg is None:
            cfg = IONetConfig(**kwargs)
        self.cfg = cfg
        c_in = cfg.in_channels * 3
        h = cfg.hidden
        self.input_proj = nn.Conv1d(c_in, h, kernel_size=1)
        self.tcn = nn.Sequential(_TCNBlock(h, 3, 1, cfg.dropout), _TCNBlock(h, 3, 2, cfg.dropout), _TCNBlock(h, 3, 4, cfg.dropout))
        self.horizon_queries = nn.Parameter(torch.randn(cfg.pred_len, h) * 0.02)
        self.horizon_attn = nn.MultiheadAttention(h, num_heads=4, dropout=cfg.dropout, batch_first=True)
        n_q = len(cfg.quantiles)
        self.quantile_head = nn.Linear(h, n_q * cfg.n_forecast)

    def _encode(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(x)
        delta = normalize_delta(time_since_last_observation(mask), x.size(-1))
        feat = torch.cat([x, mask, delta], dim=1)
        return self.tcn(self.input_proj(feat))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]=None, exo: Optional[torch.Tensor]=None) -> dict[str, torch.Tensor]:
        del exo
        cfg = self.cfg
        if mask is None:
            mask = (~torch.isnan(x)).float()
        enc = self._encode(x, mask)
        enc_t = enc.transpose(1, 2)
        q = self.horizon_queries.unsqueeze(0).expand(enc.size(0), -1, -1)
        horizon_ctx, _ = self.horizon_attn(q, enc_t, enc_t, need_weights=False)
        n_q = len(cfg.quantiles)
        qflat = self.quantile_head(horizon_ctx)
        forecast_q = qflat.view(enc.size(0), cfg.pred_len, n_q, cfg.n_forecast).permute(0, 2, 3, 1)
        if cfg.use_persist_anchor:
            do_i = min(max(cfg.do_channel_index, 0), x.size(1) - 1)
            do_last = torch.nan_to_num(x[:, do_i, -1], nan=0.0) * mask[:, do_i, -1]
            anchor = torch.zeros_like(forecast_q)
            anchor[:, :, 0, :] = do_last.unsqueeze(1).unsqueeze(-1)
            if cfg.n_forecast > cfg.chla_forecast_index:
                pass
            forecast_q = anchor + forecast_q
        med_idx = min(range(n_q), key=lambda i: abs(cfg.quantiles[i] - 0.5))
        forecast = forecast_q[:, med_idx]
        return {'forecast': forecast, 'forecast_quantiles': forecast_q, 'quantile_levels': torch.tensor(cfg.quantiles, device=x.device, dtype=x.dtype), 'moe_loss': torch.zeros((), device=x.device, dtype=x.dtype), 'soft': torch.zeros(x.size(0), 3, device=x.device, dtype=x.dtype)}

def build_ionet(**kwargs) -> IONet:
    return IONet(IONetConfig(**kwargs))

def pinball_loss(pred_q: torch.Tensor, target: torch.Tensor, quantile_levels: torch.Tensor | Sequence[float], mask: Optional[torch.Tensor]=None) -> torch.Tensor:
    if not torch.is_tensor(quantile_levels):
        q_levels = torch.tensor(quantile_levels, device=pred_q.device, dtype=pred_q.dtype)
    else:
        q_levels = quantile_levels.to(device=pred_q.device, dtype=pred_q.dtype)
    tgt = target.unsqueeze(1).expand_as(pred_q)
    qs = q_levels.view(1, pred_q.size(1), *[1] * (pred_q.dim() - 2))
    loss = torch.maximum(qs * (tgt - pred_q), (qs - 1.0) * (tgt - pred_q))
    if mask is not None:
        m = mask.unsqueeze(1).expand_as(pred_q).float()
        valid = m > 0
        loss = loss * valid.float()
        return loss.sum() / valid.float().sum().clamp(min=1.0)
    return loss.mean()

def ionet_loss(out: dict[str, torch.Tensor], forecast_y: torch.Tensor, forecast_y_mask: Optional[torch.Tensor]=None) -> dict[str, torch.Tensor]:
    l_fc = pinball_loss(out['forecast_quantiles'], forecast_y, out['quantile_levels'], forecast_y_mask)
    return {'forecast': l_fc, 'total': l_fc, 'moe_loss': out['moe_loss']}
if __name__ == '__main__':
    m = build_ionet(in_channels=5, seq_len=168, pred_len=6)
    x = torch.randn(2, 5, 168)
    mask = (torch.rand(2, 5, 168) > 0.2).float()
    o = m(x, mask)
    y = torch.randn(2, 2, 6)
    ym = torch.ones(2, 2, 6)
    loss = ionet_loss(o, y, ym)
    print(o['forecast_quantiles'].shape, float(loss['total']))
