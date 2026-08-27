from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def _gap_lengths(mask_2d: np.ndarray, max_windows: int=5000) -> list[int]:
    n = min(mask_2d.shape[0], max_windows)
    run_lengths: list[int] = []
    for w in range(n):
        missing = mask_2d[w] <= 0
        if not missing.any():
            continue
        padded = np.concatenate(([False], missing, [False]))
        changes = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        run_lengths.extend((int(e - s) for s, e in zip(starts, ends)))
    return run_lengths

def _lake_stats(proc_dir: Path) -> dict | None:
    meta_path = proc_dir / 'meta.json'
    train_path = proc_dir / 'train.npz'
    if not meta_path.exists() or not train_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    data = np.load(train_path, allow_pickle=False)
    n = int(data['x'].shape[0])
    if n == 0:
        return None
    lake_id = proc_dir.name.replace('lakebed_', '').replace('_4h', '').upper()
    easy = list(meta.get('easy_vars', []))
    soft = list(meta.get('soft_vars', []))
    miss_easy = {}
    for i, v in enumerate(easy):
        m = data['mask'][:, i, :]
        miss_easy[v] = float(1.0 - m[m >= 0].mean()) if m.size else float('nan')
    miss_soft = {}
    for i, v in enumerate(soft):
        sm = data['soft_y_mask'][:, i]
        miss_soft[v] = float(1.0 - (sm > 0).mean()) if sm.size else float('nan')
    run_lengths: list[int] = []
    mask = data['mask']
    for i in range(mask.shape[1]):
        run_lengths.extend(_gap_lengths(mask[:, i, :]))
    return {'lake_id': lake_id, 'proc_dir': str(proc_dir), 'n_train_windows': n, 'seq_len': int(meta.get('seq_len', 168)), 'missing_rate_easy': miss_easy, 'missing_rate_soft': miss_soft, 'gap_len_median_steps': float(np.median(run_lengths)) if run_lengths else float('nan'), 'gap_len_p90_steps': float(np.quantile(run_lengths, 0.9)) if run_lengths else float('nan'), 'has_soft_lab_mask': bool(meta.get('has_soft_lab_mask', False))}

def _resolve_proc_dirs(proc_glob: str) -> list[Path]:
    pattern = Path(proc_glob)
    if pattern.is_absolute():
        parent, name = (pattern.parent, pattern.name)
        candidates = parent.glob(name) if parent.exists() else []
    else:
        candidates = ROOT.glob(proc_glob)
    return sorted((p for p in candidates if p.is_dir()))

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--proc-glob', type=str, default='data/processed/lakebed_*_4h')
    ap.add_argument('--lakes', type=str, default='', help='Optional comma list, e.g. ME,BVR,...,SUGG. Empty = all matched dirs.')
    ap.add_argument('--out-json', type=Path, default=ROOT / 'results/missingness_atlas.json')
    ap.add_argument('--out-md', type=Path, default=None)
    args = ap.parse_args()
    proc_dirs = _resolve_proc_dirs(args.proc_glob)
    if args.lakes.strip():
        want = [x.strip().upper() for x in args.lakes.split(',') if x.strip()]
        by_id = {p.name.replace('lakebed_', '').replace('_4h', '').upper(): p for p in proc_dirs}
        missing = [lk for lk in want if lk not in by_id]
        if missing:
            raise SystemExit(f'[atlas] lakes not found under proc-glob: {missing}')
        proc_dirs = [by_id[lk] for lk in want]
    if not proc_dirs:
        raise SystemExit(f'[atlas] no proc dirs matched: {args.proc_glob}')
    print(f'[atlas] scanning {len(proc_dirs)} lakes ...', flush=True)
    rows = []
    for idx, p in enumerate(proc_dirs, 1):
        print(f'[atlas] ({idx}/{len(proc_dirs)}) start {p.name}', flush=True)
        st = _lake_stats(p)
        if st:
            rows.append(st)
            print(f"[atlas] ({idx}/{len(proc_dirs)}) done {st['lake_id']} train={st['n_train_windows']} chla_miss={st['missing_rate_soft'].get('chla', 'na')}", flush=True)
        else:
            print(f'[atlas] ({idx}/{len(proc_dirs)}) skip {p.name} (no train.npz/meta)', flush=True)
    payload = {'n_lakes': len(rows), 'lakes': rows}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f'[atlas] saved -> {args.out_json}')
    if args.out_md:
        lines = ['| Lake | train windows | Chl-a miss | DO miss (easy) | gap p90 (4h steps) |', '|------|---------------|------------|----------------|---------------------|']
        for r in rows:
            lines.append(f"| {r['lake_id']} | {r['n_train_windows']} | {r['missing_rate_soft'].get('chla', float('nan')):.2%} | {r['missing_rate_easy'].get('do', float('nan')):.2%} | {r['gap_len_p90_steps']:.0f} |")
        args.out_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        print(f'[atlas] table -> {args.out_md}')
if __name__ == '__main__':
    main()
