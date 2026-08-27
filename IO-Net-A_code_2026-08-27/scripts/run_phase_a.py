from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.backbone import resolve_backbone
from scripts.run_robustness import apply_linear_impute, apply_locf_impute, apply_random_mask, _copy_sidecars

def run(cmd: list[str]) -> None:
    print('+', ' '.join(cmd), flush=True)
    env = os.environ.copy()
    env.setdefault('MKL_THREADING_LAYER', 'GNU')
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
    if proc.returncode != 0:
        raise SystemExit(f"failed ({proc.returncode}): {' '.join(cmd)}")

def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))

def chla_h6_nse(path: Path) -> float | None:
    if not path.exists():
        return None
    d = _read_json(path)
    met = d.get('metrics', d)
    chla = met.get('by_target', {}).get('chla', {})
    h6 = chla.get('h6') or chla.get('6')
    if isinstance(h6, dict) and 'nse' in h6:
        return float(h6['nse'])
    return None

def chla_h6_crps(path: Path, key: str='crps') -> float | None:
    if not path.exists():
        return None
    d = _read_json(path)
    met = d.get('metrics', d)
    chla = met.get('by_target', {}).get('chla', {}).get('h6', {})
    v = chla.get(key)
    return float(v) if v is not None else None

def _done_point(out: Path) -> bool:
    return (out / 'E2_metrics.json').exists() and (out / 'best.pt').exists()

def _done_ionet_a(out: Path) -> bool:
    return (out / 'E2_crps.json').exists() and (out / 'best.pt').exists()

def train_eval_ionet_a(proc: Path, out: Path, seed: int, device: str, force: bool) -> None:
    spec = resolve_backbone('ionet_a')
    if _done_ionet_a(out) and (not force):
        print(f'[phaseA] skip ionet_a {out}', flush=True)
        return
    cmd = [PY, str(ROOT / 'scripts' / spec.train_script), '--proc-dir', str(proc), '--hyperparams', str(ROOT / 'configs' / spec.hp_file), '--seed', str(seed), '--device', device, '--out-dir', str(out)]
    if force:
        cmd.append('--force')
    run(cmd)
    run([PY, str(ROOT / 'scripts' / spec.eval_script), '--proc-dir', str(proc), '--results-dir', str(out), '--out', str(out / 'E2_crps.json'), '--denormalize', '1', '--calibrate', '0', '--device', device])

def train_eval_torch(model: str, proc: Path, out: Path, seed: int, device: str, force: bool) -> None:
    if _done_point(out) and (not force):
        print(f'[phaseA] skip {model} {out}', flush=True)
        return
    cmd = [PY, str(ROOT / 'scripts/train.py'), '--model', model, '--proc-dir', str(proc), '--hyperparams', str(ROOT / 'configs/hyperparams_locked.json'), '--seed', str(seed), '--device', device, '--out-dir', str(out)]
    if force:
        cmd.append('--force')
    run(cmd)
    run([PY, str(ROOT / 'scripts/evaluate.py'), '--exp', 'E2', '--model', model, '--results-dir', str(out), '--proc-dir', str(proc), '--denormalize', '1', '--targets', 'do,chla', '--device', device, '--out', str(out / 'E2_metrics.json')])

def prepare_mcar(src: Path, dst: Path, rate: float) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'val', 'test'):
        p = src / f'{split}.npz'
        if p.exists():
            apply_random_mask(p, dst / f'{split}.npz', rate, 42)
    _copy_sidecars(src, dst)

def prepare_impute(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'val', 'test'):
        p = src / f'{split}.npz'
        if p.exists():
            apply_linear_impute(p, dst / f'{split}.npz')
    _copy_sidecars(src, dst)

def prepare_locf(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'val', 'test'):
        p = src / f'{split}.npz'
        if p.exists():
            apply_locf_impute(p, dst / f'{split}.npz')
    _copy_sidecars(src, dst)

def summarize_nse(root: Path, tag: str, models: list[str], seeds: list[int]) -> None:
    print(f'\n=== {tag}  24h Chl-a NSE (point models) ===', flush=True)
    for model in models:
        vals = []
        for s in seeds:
            nse = chla_h6_nse(root / f'{model}_seed{s}' / 'E2_metrics.json')
            print(f'  {model} seed{s}: NSE={nse}', flush=True)
            if nse is not None:
                vals.append(nse)
        if vals:
            mu = sum(vals) / len(vals)
            sd = (sum(((x - mu) ** 2 for x in vals)) / max(len(vals) - 1, 1)) ** 0.5
            print(f'  {model} NSE mean±std = {mu:.4f} ± {sd:.4f}  n={len(vals)}', flush=True)

def summarize_ionet_a(root: Path, tag: str, seeds: list[int]) -> None:
    print(f'\n=== {tag}  IO-Net-A 24h Chl-a CRPS ===', flush=True)
    crps_vals, dcrps_vals = ([], [])
    for s in seeds:
        p = root / f'ionet_a_seed{s}' / 'E2_crps.json'
        crps = chla_h6_crps(p, 'crps')
        dcrps = chla_h6_crps(p, 'delta_crps_vs_persistence')
        print(f'  ionet_a seed{s}: CRPS={crps} ΔCRPS={dcrps}', flush=True)
        if crps is not None:
            crps_vals.append(crps)
        if dcrps is not None:
            dcrps_vals.append(dcrps)
    if crps_vals:
        mu = sum(crps_vals) / len(crps_vals)
        print(f'  ionet_a CRPS mean = {mu:.4f}  n={len(crps_vals)}', flush=True)
    if dcrps_vals:
        mu = sum(dcrps_vals) / len(dcrps_vals)
        print(f'  ionet_a ΔCRPS mean = {mu:.4f}  n={len(dcrps_vals)}', flush=True)

def main() -> None:
    ap = argparse.ArgumentParser(description='NW Phase A (')
    ap.add_argument('--proc-dir', type=Path, default=ROOT / 'data/processed/lakebed_BVR_4h')
    ap.add_argument('--results-root', type=Path, default=ROOT / 'results/BVR_4h/phaseA')
    ap.add_argument('--seeds', default='0,1,2,3,4,5,6,7,8,9')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--skip', default='', help='comma: native_ionet_a,native_lstm,native_grud,mcar_ionet_a,mcar_lstm,mcar_grud,impute_lstm,locf_lstm')
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    skip = {x.strip() for x in args.skip.split(',') if x.strip()}
    proc = args.proc_dir
    root = args.results_root
    native = root / 'native'
    mcar_dir = root / 'mcar03'
    impute_dir = root / 'impute'
    locf_dir = root / 'locf'
    mcar_proc = root / 'data_mcar03'
    impute_proc = root / 'data_impute'
    locf_proc = root / 'data_locf'
    native.mkdir(parents=True, exist_ok=True)
    if 'native_ionet_a' not in skip:
        for s in seeds:
            train_eval_ionet_a(proc, native / f'ionet_a_seed{s}', s, args.device, args.force)
    if 'native_lstm' not in skip:
        for s in seeds:
            train_eval_torch('lstm', proc, native / f'lstm_seed{s}', s, args.device, args.force)
    if 'native_grud' not in skip:
        for s in seeds:
            train_eval_torch('grud', proc, native / f'grud_seed{s}', s, args.device, args.force)
    summarize_ionet_a(native, 'Native', seeds)
    summarize_nse(native, 'Native', ['lstm', 'grud'], seeds)
    if 'mcar_ionet_a' not in skip or 'mcar_lstm' not in skip or 'mcar_grud' not in skip:
        if not (mcar_proc / 'train.npz').exists():
            prepare_mcar(proc, mcar_proc, 0.3)
        mcar_dir.mkdir(parents=True, exist_ok=True)
        if 'mcar_ionet_a' not in skip:
            for s in seeds:
                train_eval_ionet_a(mcar_proc, mcar_dir / f'ionet_a_seed{s}', s, args.device, args.force)
        if 'mcar_lstm' not in skip:
            for s in seeds:
                train_eval_torch('lstm', mcar_proc, mcar_dir / f'lstm_seed{s}', s, args.device, args.force)
        if 'mcar_grud' not in skip:
            for s in seeds:
                train_eval_torch('grud', mcar_proc, mcar_dir / f'grud_seed{s}', s, args.device, args.force)
        summarize_ionet_a(mcar_dir, 'MCAR-0.3', seeds)
        summarize_nse(mcar_dir, 'MCAR-0.3', ['lstm', 'grud'], seeds)
    if 'impute_lstm' not in skip:
        if not (impute_proc / 'train.npz').exists():
            prepare_impute(proc, impute_proc)
        impute_dir.mkdir(parents=True, exist_ok=True)
        for s in seeds:
            train_eval_torch('lstm', impute_proc, impute_dir / f'lstm_seed{s}', s, args.device, args.force)
        summarize_nse(impute_dir, 'LSTM+linear-impute', ['lstm'], seeds)
    if 'locf_lstm' not in skip:
        if not (locf_proc / 'train.npz').exists():
            prepare_locf(proc, locf_proc)
        locf_dir.mkdir(parents=True, exist_ok=True)
        for s in seeds:
            train_eval_torch('lstm', locf_proc, locf_dir / f'lstm_seed{s}', s, args.device, args.force)
        summarize_nse(locf_dir, 'LSTM+LOCF-impute', ['lstm'], seeds)
    print('\n[phaseA-ionet] done. Gate A: impute/LOCF LSTM NSE << native LSTM; IO-Net-A / GRU-D use explicit missingness (no other models in this paper).', flush=True)
if __name__ == '__main__':
    main()
