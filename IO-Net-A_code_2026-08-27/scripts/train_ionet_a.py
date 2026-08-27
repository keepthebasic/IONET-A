from __future__ import annotations
import os
os.environ.setdefault('MKL_THREADING_LAYER', 'GNU')
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from models.ionet_a import IONetAConfig, build_ionet_a, ionet_a_loss
from scripts.experiment_utils import set_seed
from scripts.npz_dataset import NPZDataset, load_meta, split_npz_paths
from scripts.train import _json_safe, apply_hyperparams_to_train_args, load_hyperparams, subsample_indices
MODEL_NAME = 'ionet_a'

def pick_device(name: str) -> torch.device:
    if name in ('cuda', 'auto', 'gpu') and (not torch.cuda.is_available()):
        print(f'[train_{MODEL_NAME}] WARNING: cuda unavailable; using cpu')
        return torch.device('cpu')
    return torch.device('cuda' if name == 'cuda' and torch.cuda.is_available() else 'cpu')

def _forward_batch(model: nn.Module, batch: dict[str, torch.Tensor], device: torch.device):
    return model(batch['x'].to(device), batch['mask'].to(device), None, soft_y=batch['soft_y'].to(device), soft_y_mask=batch['soft_y_mask'].to(device), soft_lab_mask=batch['soft_lab_mask'].to(device) if 'soft_lab_mask' in batch else None)

def run_epoch(model: nn.Module, loader: DataLoader, device: torch.device, optimizer: torch.optim.Optimizer | None, train: bool, cfg: IONetAConfig, chla_weight: float) -> dict[str, float]:
    model.train(train)
    totals = {'total': 0.0, 'forecast': 0.0, 'width': 0.0}
    n_batches = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            if optimizer:
                optimizer.zero_grad(set_to_none=True)
            out = _forward_batch(model, batch, device)
            losses = ionet_a_loss(out, batch['forecast_y'].to(device), batch['forecast_y_mask'].to(device), cfg, chla_weight=chla_weight)
            if train:
                losses['total'].backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            for k in totals:
                if k in losses:
                    totals[k] += float(losses[k].detach().cpu())
            n_batches += 1
    return {k: v / max(n_batches, 1) for k, v in totals.items()}

def main() -> None:
    ap = argparse.ArgumentParser(description='Train IO-Net-A (Assay-anchored IO-Net)')
    ap.add_argument('--proc-dir', type=Path, required=True)
    ap.add_argument('--hyperparams', type=Path, default=ROOT / 'configs' / 'hyperparams_ionet_a.json')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=0.0008)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--train-frac', type=float, default=1.0)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    hp = load_hyperparams(args.hyperparams, MODEL_NAME)
    apply_hyperparams_to_train_args(args, hp)
    chla_weight = float(hp.get('chla_pinball_weight', 1.5))
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = load_meta(args.proc_dir)
    with open(args.out_dir / 'run_config.json', 'w', encoding='utf-8') as f:
        json.dump({'model': MODEL_NAME, 'proc_dir': str(args.proc_dir), 'seed': args.seed, 'hyperparams': hp}, f, indent=2, default=str)
    ckpt_path = args.out_dir / 'best.pt'
    if ckpt_path.exists() and (not args.force):
        try:
            old = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            if old.get('model') not in ('ionet_a', 'ionet_lite'):
                print(f"[train_{MODEL_NAME}] stale ckpt model={old.get('model')!r}; retrain with --force")
            else:
                print(f'[train_{MODEL_NAME}] skip existing -> {ckpt_path}')
                return
        except Exception:
            print(f'[train_{MODEL_NAME}] unreadable ckpt; retrain -> {ckpt_path}')
    device = pick_device(args.device)
    paths = split_npz_paths(args.proc_dir)
    train_idx = subsample_indices(len(NPZDataset(paths['train'])), args.train_frac, args.seed)
    train_loader = DataLoader(NPZDataset(paths['train'], train_idx), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(NPZDataset(paths['val']), batch_size=args.batch_size, shuffle=False)
    model = build_ionet_a(in_channels=len(meta['easy_vars']), seq_len=int(meta['seq_len']), pred_len=int(meta['pred_len']), n_forecast=len(meta['forecast_vars']), hidden=int(args.hidden), dropout=float(args.dropout), width_reg_weight=float(hp.get('width_reg_weight', 0.1))).to(device)
    cfg = model.cfg
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0001)
    best_val, best_epoch, stale = (float('inf'), -1, 0)
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, device, optimizer, True, cfg, chla_weight)
        va = run_epoch(model, val_loader, device, None, False, cfg, chla_weight)
        history.append({'epoch': epoch, 'train': tr, 'val': va, 'sec': time.time() - t0})
        print(f"[train_{MODEL_NAME}] epoch {epoch:03d} | train={tr['total']:.4f} val={va['total']:.4f} width={va['width']:.4f}")
        if va['total'] < best_val:
            best_val, best_epoch, stale = (va['total'], epoch, 0)
            torch.save({'model': MODEL_NAME, 'state_dict': model.state_dict(), 'meta': meta, 'args': _json_safe(vars(args)), 'ionet_a_cfg': _json_safe(vars(model.cfg)), 'best_val': best_val, 'epoch': epoch}, ckpt_path)
        else:
            stale += 1
            if stale >= args.patience:
                print(f'[train_{MODEL_NAME}] early stop @ {epoch} (best={best_val:.4f} ep {best_epoch})')
                break
    (args.out_dir / 'history.json').write_text(json.dumps(_json_safe(history), indent=2), encoding='utf-8')
    print(f'[train_{MODEL_NAME}] saved -> {ckpt_path}')
if __name__ == '__main__':
    main()
