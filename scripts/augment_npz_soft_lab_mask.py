from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT / 'scripts'))
from common import EASY_VARS, SOFT_VARS
from data_process import _blocked_split, _invalidate_constant_runs, _load_csv

def _soft_lab_windows(df, seq_len: int, pred_len: int, stride: int, row_slice: slice) -> np.ndarray:
    sub = df.iloc[row_slice].reset_index(drop=True)
    n_rows = len(sub)
    need = seq_len + pred_len
    soft_mask = np.stack([sub[f'{v}_mask'].to_numpy(dtype=np.float32) for v in SOFT_VARS], axis=0)
    packs = []
    for start in range(0, n_rows - need + 1, stride):
        end = start + seq_len
        packs.append(soft_mask[:, start:end])
    if not packs:
        return np.zeros((0, len(SOFT_VARS), seq_len), dtype=np.float32)
    return np.stack(packs, axis=0)

def _resolve_csv(meta: dict, proc_dir: Path) -> Path:
    raw = str(meta.get('input_csv', '')).strip().replace('\\', '/')
    candidates: list[Path] = []
    if raw:
        p = Path(raw)
        candidates.append(p if p.is_absolute() else ROOT / p)
        candidates.append(proc_dir / Path(raw).name)
    candidates.append(proc_dir / f"{proc_dir.name.replace('lakebed_', '').replace('_4h', '')}_4h.csv")
    candidates.append(proc_dir / 'BVR_4h.csv')
    for c in candidates:
        if c.exists():
            return c
    tried = '\n  '.join((str(c) for c in candidates))
    raise SystemExit(f'CSV not found. Tried:\n  {tried}')

def augment_proc_dir(proc_dir: Path, force: bool=False) -> None:
    meta = json.loads((proc_dir / 'meta.json').read_text(encoding='utf-8'))
    csv_path = _resolve_csv(meta, proc_dir)
    print(f'[augment] using CSV: {csv_path}')
    seq_len = int(meta['seq_len'])
    pred_len = int(meta['pred_len'])
    stride = int(meta.get('stride', 1))
    split = tuple((float(x) for x in meta['split']))
    df = _load_csv(csv_path)
    qc_cols = [c for c in list(EASY_VARS) + list(SOFT_VARS) if c in df.columns]
    df = _invalidate_constant_runs(df, qc_cols, min_run=36)
    train_sl, val_sl, test_sl = _blocked_split(len(df), split)
    splits = {'train': train_sl, 'val': val_sl, 'test': test_sl}
    for name, sl in splits.items():
        npz_path = proc_dir / f'{name}.npz'
        if not npz_path.exists():
            print(f'[augment] skip missing {npz_path}')
            continue
        data = dict(np.load(npz_path, allow_pickle=False))
        if 'soft_lab_mask' in data and (not force):
            print(f"[augment] {name}: soft_lab_mask already present (n={data['soft_lab_mask'].shape})")
            continue
        lab = _soft_lab_windows(df, seq_len, pred_len, stride, sl)
        n_x = int(data['x'].shape[0])
        if lab.shape[0] != n_x:
            raise SystemExit(f'[augment] {name}: window count mismatch lab={lab.shape[0]} vs x={n_x}')
        data['soft_lab_mask'] = lab
        np.savez_compressed(npz_path, **data)
        hit = float((lab.max(axis=(1, 2)) > 0).mean()) if n_x else 0.0
        print(f'[augment] {name}: soft_lab_mask {lab.shape} windows_with_any_lab={hit:.3f} -> {npz_path}')
    meta['has_soft_lab_mask'] = True
    meta['soft_lab_mask_vars'] = list(SOFT_VARS)
    (proc_dir / 'meta.json').write_text(json.dumps(meta, indent=2, default=str), encoding='utf-8')
    print(f"[augment] meta updated -> {proc_dir / 'meta.json'}")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--proc-dir', type=Path, required=True)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    augment_proc_dir(args.proc_dir, force=args.force)
if __name__ == '__main__':
    main()
