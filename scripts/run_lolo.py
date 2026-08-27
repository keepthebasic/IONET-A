from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.augment_npz_soft_lab_mask import augment_proc_dir
from scripts.npz_dataset import build_lolo_proc_dir

def run(cmd: list[str]) -> None:
    print('+', ' '.join(cmd), flush=True)
    env = os.environ.copy()
    env.setdefault('MKL_THREADING_LAYER', 'GNU')
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
    if proc.returncode != 0:
        raise SystemExit(f"failed ({proc.returncode}): {' '.join(cmd)}")

def _crps_h6(path: Path, target: str='chla') -> float | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding='utf-8'))
    h = d.get('metrics', {}).get('by_target', {}).get(target, {}).get('h6', {})
    v = h.get('crps')
    return float(v) if v is not None else None

def _delta_crps_h6(path: Path, target: str='chla') -> float | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding='utf-8'))
    h = d.get('metrics', {}).get('by_target', {}).get(target, {}).get('h6', {})
    v = h.get('delta_crps_vs_persistence')
    return float(v) if v is not None else None

def _ensure_source_lakes(proc_root: Path, lakes: list[str], augment: bool) -> None:
    for lk in lakes:
        proc = proc_root / f'lakebed_{lk}_4h'
        if not (proc / 'train.npz').exists():
            raise SystemExit(f"[lolo] missing {proc / 'train.npz'}")
        if augment:
            augment_proc_dir(proc, force=False)

def main() -> None:
    ap = argparse.ArgumentParser(description='NW LOLO with IO-Net-A')
    ap.add_argument('--proc-root', type=Path, default=ROOT / 'data/processed')
    ap.add_argument('--out-dir', type=Path, default=ROOT / 'results/BVR_4h/lolo')
    ap.add_argument('--lakes', default='ME,BVR,FCR,TR,SP')
    ap.add_argument('--seeds', default='0,1,2,3,4')
    ap.add_argument('--scaler-mode', default='target_train_only', choices=['target_train_only', 'train_lakes'])
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--augment-sources', action='store_true', help='augment each lake NPZ before LOLO merge')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    lakes = [x.strip() for x in args.lakes.split(',') if x.strip()]
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[lolo-ionet] lakes={lakes} seeds={seeds}', flush=True)
    _ensure_source_lakes(args.proc_root, lakes, args.augment_sources)
    summary: dict = {'lakes': lakes, 'seeds': seeds, 'folds': {}}
    for test_lake in lakes:
        train_lakes = [lk for lk in lakes if lk != test_lake]
        fold_dir = args.out_dir / f'test_{test_lake}'
        proc_dir = fold_dir / '_proc'
        fold_dir.mkdir(parents=True, exist_ok=True)
        if not (proc_dir / 'meta.json').exists() or args.force:
            build_lolo_proc_dir(args.proc_root, train_lakes, test_lake, proc_dir, args.scaler_mode)
        if not (proc_dir / 'meta.json').exists():
            raise SystemExit(f'[lolo] build failed: no meta.json in {proc_dir}')
        fold_summary: dict = {'train_lakes': train_lakes, 'test_lake': test_lake, 'proc_dir': str(proc_dir), 'seeds': {}}
        for seed in seeds:
            run_dir = fold_dir / f'ionet_a_seed{seed}'
            ckpt = run_dir / 'best.pt'
            out_json = run_dir / 'E2_crps.json'
            if not ckpt.exists() or args.force:
                run([PY, str(ROOT / 'scripts/train_ionet_a.py'), '--proc-dir', str(proc_dir), '--hyperparams', str(ROOT / 'configs/hyperparams_ionet_a.json'), '--seed', str(seed), '--device', args.device, '--out-dir', str(run_dir)] + (['--force'] if args.force else []))
            if not out_json.exists() or args.force:
                run([PY, str(ROOT / 'scripts/evaluate_ionet_a.py'), '--proc-dir', str(proc_dir), '--results-dir', str(run_dir), '--out', str(out_json), '--denormalize', '1', '--calibrate', '0', '--device', args.device])
            fold_summary['seeds'][str(seed)] = {'chla_crps': _crps_h6(out_json, 'chla'), 'chla_delta_crps': _delta_crps_h6(out_json, 'chla'), 'do_crps': _crps_h6(out_json, 'do'), 'do_delta_crps': _delta_crps_h6(out_json, 'do')}
            print(f"[lolo-ionet] test={test_lake} seed={seed} Chl-a CRPS={fold_summary['seeds'][str(seed)]['chla_crps']} dCRPS={fold_summary['seeds'][str(seed)]['chla_delta_crps']}", flush=True)
        chla = [v['chla_crps'] for v in fold_summary['seeds'].values() if v['chla_crps'] is not None]
        if chla:
            fold_summary['chla_crps_mean'] = float(np.mean(chla))
            fold_summary['chla_crps_std'] = float(np.std(chla, ddof=1)) if len(chla) > 1 else 0.0
        summary['folds'][test_lake] = fold_summary
    out_path = args.out_dir / 'lolo_summary.json'
    out_path.write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')
    print(f'\n[lolo-ionet] done -> {out_path}', flush=True)
    print('\n=== LOLO summary (24h Chl-a CRPS, mean over seeds) ===', flush=True)
    for test_lake, fold in summary['folds'].items():
        mu = fold.get('chla_crps_mean')
        sd = fold.get('chla_crps_std', 0.0)
        if mu is not None:
            print(f'  held-out {test_lake}: {mu:.4f} ± {sd:.4f}', flush=True)
if __name__ == '__main__':
    main()
