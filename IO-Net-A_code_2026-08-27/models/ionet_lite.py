from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ionet import DEFAULT_QUANTILES, _TCNBlock, normalize_delta, pinball_loss, time_since_last_observation

@dataclass
class IONetLiteConfig:
    in_channels: int = 5
    seq_len: int = 168
    pred_len: int = 6
    n_forecast: int = 2
    hidden: int = 64
    dropout: float = 0.1
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES
    do_channel_index: int = 4
    chla_forecast_index: int = 1
    chla_soft_index: int = 2
    use_persist_anchor: bool = True
    use_diurnal_template: bool = True
    use_mask_gate: bool = True
    use_lab_age: bool = True
    use_monotonic_quantiles: bool = True
    diurnal_period: int = 6
    width_reg_weight: float = 0.1
    min_quantile_width: float = 0.05

class IONetLite(nn.Module):

    def __init__(self, cfg: IONetLiteConfig | None=None, **kwargs):
        super().__init__()
        if cfg is None:
            cfg = IONetLiteConfig(**kwargs)
        self.cfg = cfg
        extra_in = 1 if cfg.use_lab_age else 0
        c_in = cfg.in_channels * 3 + extra_in
        h = cfg.hidden
        self.input_proj = nn.Conv1d(c_in, h, kernel_size=1)
        self.tcn = nn.Sequential(_TCNBlock(h, 3, 1, cfg.dropout), _TCNBlock(h, 3, 2, cfg.dropout), _TCNBlock(h, 3, 4, cfg.dropout))
        if cfg.use_mask_gate:
            self.mask_gate = nn.Linear(cfg.in_channels, h)
        else:
            self.mask_gate = None
        if cfg.use_diurnal_template:
            self.diurnal = nn.Parameter(torch.zeros(cfg.in_channels, cfg.diurnal_period))
        else:
            self.diurnal = None
        self.horizon_queries = nn.Parameter(torch.randn(cfg.pred_len, h) * 0.02)
        self.horizon_attn = nn.MultiheadAttention(h, num_heads=4, dropout=cfg.dropout, batch_first=True)
        if cfg.use_monotonic_quantiles:
            self.delta_head = nn.Linear(h, cfg.n_forecast)
            self.width_head = nn.Linear(h, cfg.n_forecast * 2)
        else:
            n_q = len(cfg.quantiles)
            self.quantile_head = nn.Linear(h, n_q * cfg.n_forecast)

    @property
    def uses_soft_context(self) -> bool:
        return True

    def _apply_diurnal(self, x: torch.Tensor) -> torch.Tensor:
        if self.diurnal is None:
            return x
        length = x.size(-1)
        phase = torch.arange(length, device=x.device) % self.cfg.diurnal_period
        template = self.diurnal[:, phase]
        return x - template.unsqueeze(0)

    def _lab_age_channel(self, soft_lab_mask: Optional[torch.Tensor], length: int, device: torch.device, dtype: torch.dtype, batch: int) -> torch.Tensor:
        if not self.cfg.use_lab_age or soft_lab_mask is None:
            return torch.zeros(batch, 1, length, device=device, dtype=dtype)
        chla_i = min(max(self.cfg.chla_soft_index, 0), soft_lab_mask.size(1) - 1)
        lab_m = soft_lab_mask[:, chla_i:chla_i + 1, :length]
        age = time_since_last_observation(lab_m)
        return normalize_delta(age, length)

    def _encode(self, x: torch.Tensor, mask: torch.Tensor, soft_lab_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(x)
        x_enc = self._apply_diurnal(x)
        delta = normalize_delta(time_since_last_observation(mask), x.size(-1))
        parts = [x_enc, mask, delta]
        if self.cfg.use_lab_age:
            parts.append(self._lab_age_channel(soft_lab_mask, x.size(-1), x.device, x.dtype, x.size(0)))
        feat = torch.cat(parts, dim=1)
        enc = self.tcn(self.input_proj(feat))
        if self.mask_gate is not None:
            gate = torch.sigmoid(self.mask_gate(mask.mean(dim=-1)))
            enc = enc * gate.unsqueeze(-1)
        return enc

    def _build_anchors(self, x: torch.Tensor, mask: torch.Tensor, soft_y: Optional[torch.Tensor], soft_y_mask: Optional[torch.Tensor], batch: int, pred_len: int, n_forecast: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        cfg = self.cfg
        anchor = torch.zeros(batch, len(cfg.quantiles), n_forecast, pred_len, device=device, dtype=dtype)
        if not cfg.use_persist_anchor:
            return anchor
        do_i = min(max(cfg.do_channel_index, 0), x.size(1) - 1)
        do_last = torch.nan_to_num(x[:, do_i, -1], nan=0.0) * mask[:, do_i, -1]
        n_q = len(cfg.quantiles)
        anchor[:, :, 0, :] = do_last.view(batch, 1, 1).expand(batch, n_q, pred_len)
        if n_forecast > cfg.chla_forecast_index and soft_y is not None:
            chla_i = min(max(cfg.chla_soft_index, 0), soft_y.size(1) - 1)
            chla_last = soft_y[:, chla_i]
            if soft_y_mask is not None:
                chla_last = chla_last * soft_y_mask[:, chla_i]
            anchor[:, :, cfg.chla_forecast_index, :] = chla_last.view(batch, 1, 1).expand(batch, n_q, pred_len)
        return anchor

    def _quantiles_from_context(self, horizon_ctx: torch.Tensor, anchor: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        cfg = self.cfg
        b, tlen, _ = horizon_ctx.shape
        n_q = len(cfg.quantiles)
        med_idx = min(range(n_q), key=lambda i: abs(cfg.quantiles[i] - 0.5))
        if cfg.use_monotonic_quantiles:
            delta50 = self.delta_head(horizon_ctx).permute(0, 2, 1)
            widths = F.softplus(self.width_head(horizon_ctx)).view(b, tlen, cfg.n_forecast, 2)
            q50 = anchor[:, med_idx] + delta50
            lo = widths[..., 0].permute(0, 2, 1)
            hi = widths[..., 1].permute(0, 2, 1)
            q10 = q50 - lo
            q90 = q50 + hi
            forecast_q = torch.stack([q10, q50, q90], dim=1)
            width_sum = lo + hi
            return (forecast_q, width_sum)
        qflat = self.quantile_head(horizon_ctx)
        forecast_q = qflat.view(b, tlen, n_q, cfg.n_forecast).permute(0, 2, 3, 1)
        forecast_q = anchor + forecast_q
        return (forecast_q, None)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]=None, exo: Optional[torch.Tensor]=None, soft_y: Optional[torch.Tensor]=None, soft_y_mask: Optional[torch.Tensor]=None, soft_lab_mask: Optional[torch.Tensor]=None) -> dict[str, torch.Tensor]:
        del exo
        cfg = self.cfg
        if mask is None:
            mask = (~torch.isnan(x)).float()
        enc = self._encode(x, mask, soft_lab_mask)
        enc_t = enc.transpose(1, 2)
        q = self.horizon_queries.unsqueeze(0).expand(enc.size(0), -1, -1)
        horizon_ctx, _ = self.horizon_attn(q, enc_t, enc_t, need_weights=False)
        anchor = self._build_anchors(x, mask, soft_y, soft_y_mask, enc.size(0), cfg.pred_len, cfg.n_forecast, x.device, x.dtype)
        forecast_q, width_sum = self._quantiles_from_context(horizon_ctx, anchor)
        med_idx = min(range(len(cfg.quantiles)), key=lambda i: abs(cfg.quantiles[i] - 0.5))
        forecast = forecast_q[:, med_idx]
        out: dict[str, torch.Tensor] = {'forecast': forecast, 'forecast_quantiles': forecast_q, 'quantile_levels': torch.tensor(cfg.quantiles, device=x.device, dtype=x.dtype), 'moe_loss': torch.zeros((), device=x.device, dtype=x.dtype), 'soft': torch.zeros(x.size(0), 3, device=x.device, dtype=x.dtype)}
        if width_sum is not None:
            out['quantile_widths'] = width_sum
        return out

def build_ionet_lite(**kwargs) -> IONetLite:
    return IONetLite(IONetLiteConfig(**kwargs))

def ionet_lite_loss(out: dict[str, torch.Tensor], forecast_y: torch.Tensor, forecast_y_mask: Optional[torch.Tensor]=None, cfg: IONetLiteConfig | None=None, chla_weight: float=1.5, chla_forecast_index: int=1) -> dict[str, torch.Tensor]:
    l_fc = pinball_loss(out['forecast_quantiles'], forecast_y, out['quantile_levels'], forecast_y_mask)
    if forecast_y_mask is not None and forecast_y.size(1) > chla_forecast_index:
        chla_only = forecast_y_mask.new_zeros(forecast_y_mask.shape)
        chla_only[:, chla_forecast_index, :] = forecast_y_mask[:, chla_forecast_index, :]
        l_chla = pinball_loss(out['forecast_quantiles'], forecast_y, out['quantile_levels'], chla_only)
        l_fc = l_fc + (chla_weight - 1.0) * l_chla
    l_width = torch.zeros((), device=forecast_y.device, dtype=forecast_y.dtype)
    if 'quantile_widths' in out and cfg is not None:
        min_w = float(cfg.min_quantile_width)
        l_width = F.relu(min_w - out['quantile_widths']).mean()
    total = l_fc + float(cfg.width_reg_weight if cfg else 0.1) * l_width
    return {'forecast': l_fc, 'width': l_width, 'total': total, 'moe_loss': out['moe_loss']}
if __name__ == '__main__':
    m = build_ionet_lite(in_channels=5, seq_len=168, pred_len=6)
    x = torch.randn(2, 5, 168)
    mask = (torch.rand(2, 5, 168) > 0.2).float()
    soft_y = torch.randn(2, 3)
    soft_y_mask = (torch.rand(2, 3) > 0.3).float()
    soft_lab = (torch.rand(2, 3, 168) > 0.9).float()
    o = m(x, mask, soft_y=soft_y, soft_y_mask=soft_y_mask, soft_lab_mask=soft_lab)
    y = torch.randn(2, 2, 6)
    ym = torch.ones(2, 2, 6)
    loss = ionet_lite_loss(o, y, ym, m.cfg)
    print(o['forecast_quantiles'].shape, float(loss['total']))
