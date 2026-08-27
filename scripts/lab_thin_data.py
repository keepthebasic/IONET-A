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

def _copy_sidecars(src: Path, dst: Path) -> None:
    for name in ('meta.json', 'scalers.json'):
        sp = src / name
        if sp.exists():
            shutil.copy2(sp, dst / name)
    for csv in src.glob('*_4h.csv'):
        shutil.copy2(csv, dst / csv.name)

def thin_lab_train_npz(train_path: Path, out_path: Path, ratio: float, seed: int, chla_fc_idx: int=1) -> dict[str, float]:
    data = {k: np.array(v) for k, v in np.load(train_path, allow_pickle=False).items()}
    if ratio >= 1.0:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **data)
        return {'ratio': ratio, 'soft_y_retained': 1.0, 'lab_events_retained': 1.0}
    rng = np.random.default_rng(seed)
    soft_m = data['soft_y_mask'].copy()
    fc_m = data['forecast_y_mask'].copy()
    lab_m = data.get('soft_lab_mask')
    n_soft_kept, n_soft_total = (0, 0)
    for n in range(soft_m.shape[0]):
        for s in range(soft_m.shape[1]):
            if soft_m[n, s] > 0:
                n_soft_total += 1
                if rng.random() <= ratio:
                    n_soft_kept += 1
                else:
                    soft_m[n, s] = 0.0
    n_lab_kept, n_lab_total = (0, 0)
    if lab_m is not None:
        lab_m = lab_m.copy()
        for n in range(lab_m.shape[0]):
            for s in range(lab_m.shape[1]):
                pos = np.where(lab_m[n, s] > 0)[0]
                for _ in pos:
                    n_lab_total += 1
                for p in pos:
                    if rng.random() <= ratio:
                        n_lab_kept += 1
                    else:
                        lab_m[n, s, p] = 0.0
        data['soft_lab_mask'] = lab_m
    for n in range(fc_m.shape[0]):
        for h in range(fc_m.shape[2]):
            if fc_m[n, chla_fc_idx, h] > 0:
                if rng.random() > ratio:
                    fc_m[n, chla_fc_idx, h] = 0.0
    data['soft_y_mask'] = soft_m
    data['forecast_y_mask'] = fc_m
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **data)
    return {'ratio': ratio, 'soft_y_retained': n_soft_kept / n_soft_total if n_soft_total else float('nan'), 'lab_events_retained': n_lab_kept / n_lab_total if n_lab_total else float('nan')}

def prepare_thinned_proc(src: Path, dst: Path, ratio: float, seed: int, forecast_vars: list[str]) -> dict[str, float]:
    dst.mkdir(parents=True, exist_ok=True)
    chla_fc_idx = forecast_vars.index('chla') if 'chla' in forecast_vars else 1
    stats = thin_lab_train_npz(src / 'train.npz', dst / 'train.npz', ratio, seed, chla_fc_idx)
    for split in ('val', 'test'):
        sp = src / f'{split}.npz'
        if sp.exists():
            shutil.copy2(sp, dst / f'{split}.npz')
    _copy_sidecars(src, dst)
    meta = json.loads((dst / 'meta.json').read_text(encoding='utf-8'))
    meta['lab_thin_ratio'] = ratio
    meta['lab_thin_seed'] = seed
    (dst / 'meta.json').write_text(json.dumps(meta, indent=2, default=str), encoding='utf-8')
    return stats

def main() -> None:
    ap = argparse.ArgumentParser(description='Build thinned-train proc dir for lab thinning experiment')
    ap.add_argument('--proc-dir', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--ratio', type=float, required=True)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    meta = json.loads((args.proc_dir / 'meta.json').read_text(encoding='utf-8'))
    stats = prepare_thinned_proc(args.proc_dir, args.out_dir, args.ratio, args.seed, list(meta.get('forecast_vars', ['do', 'chla'])))
    print(f'[lab_thin_data] ratio={args.ratio} -> {args.out_dir} stats={stats}')
if __name__ == '__main__':
    main()
