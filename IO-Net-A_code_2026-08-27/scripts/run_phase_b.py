from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.backbone import MAIN_BACKBONE, resolve_backbone
PY = sys.executable
_WIN_SOFT_FAIL = {3221226505, 3221225477}

def run(cmd: list[str], *, ok_if: Path | None=None) -> None:
    print('+', ' '.join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if proc.returncode == 0:
        return
    if ok_if is not None and ok_if.exists() and (proc.returncode in _WIN_SOFT_FAIL):
        print(f'[phaseB] WARN: process exit {proc.returncode} but {ok_if.name} exists — continue', flush=True)
        return
    raise SystemExit(f"failed ({proc.returncode}): {' '.join(cmd)}")

def _done(out: Path, need_ckpt: bool=True) -> bool:
    if not (out / 'E2_crps.json').exists():
        return False
    if need_ckpt and (not (out / 'best.pt').exists()):
        return need_ckpt is False or (out / 'run_config.json').exists()
    return True

def chla_h6_crps(path: Path, key: str='crps') -> float | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding='utf-8'))
    chla = d.get('metrics', {}).get('by_target', {}).get('chla', {}).get('h6', {})
    v = chla.get(key)
    return float(v) if v is not None else None

def train_eval_ionet_a(proc: Path, out: Path, seed: int, device: str, force: bool) -> None:
    spec = resolve_backbone(MAIN_BACKBONE)
    if _done(out, True) and (not force):
        print(f'[phaseB] skip {spec.result_tag} {out}', flush=True)
        return
    cmd = [PY, str(ROOT / 'scripts' / spec.train_script), '--proc-dir', str(proc), '--hyperparams', str(ROOT / 'configs' / spec.hp_file), '--seed', str(seed), '--device', device, '--out-dir', str(out)]
    if force:
        cmd.append('--force')
    run(cmd)
    run([PY, str(ROOT / 'scripts' / spec.eval_script), '--proc-dir', str(proc), '--results-dir', str(out), '--out', str(out / 'E2_crps.json'), '--denormalize', '1', '--calibrate', '0', '--device', device])

def train_eval_baseline(model: str, proc: Path, out: Path, seed: int, device: str, force: bool) -> None:
    if _done(out, True) and (not force):
        print(f'[phaseB] skip {model} {out}', flush=True)
        return
    ckpt = out / 'best.pt'
    if force or not ckpt.exists():
        cmd = [PY, str(ROOT / 'scripts/train.py'), '--model', model, '--proc-dir', str(proc), '--hyperparams', str(ROOT / 'configs/hyperparams_locked.json'), '--seed', str(seed), '--device', device, '--out-dir', str(out)]
        if force:
            cmd.append('--force')
        run(cmd, ok_if=ckpt)
    else:
        print(f'[phaseB] reuse ckpt {ckpt}', flush=True)
    if not ckpt.exists():
        raise SystemExit(f'[phaseB] missing checkpoint after train: {ckpt}')
    run([PY, str(ROOT / 'scripts/evaluate_crps.py'), '--model', model, '--proc-dir', str(proc), '--results-dir', str(out), '--out', str(out / 'E2_crps.json'), '--denormalize', '1', '--device', device], ok_if=out / 'E2_crps.json')

def eval_persistence(proc: Path, out: Path, seed: int, force: bool) -> None:
    if (out / 'E2_crps.json').exists() and (not force):
        print(f'[phaseB] skip persistence {out}', flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    run([PY, str(ROOT / 'scripts/evaluate_crps.py'), '--model', 'persistence', '--proc-dir', str(proc), '--results-dir', str(out), '--seed', str(seed), '--out', str(out / 'E2_crps.json'), '--denormalize', '1'])

def summarize_crps(root: Path, models: list[str], seeds: list[int]) -> None:
    print('\n=== Phase B CRPS  24h Chl-a ===', flush=True)
    for model in models:
        vals, deltas = ([], [])
        for s in seeds:
            p = root / f'{model}_seed{s}' / 'E2_crps.json'
            crps = chla_h6_crps(p, 'crps')
            dcrps = chla_h6_crps(p, 'delta_crps_vs_persistence')
            cov = None
            if p.exists():
                chla = json.loads(p.read_text(encoding='utf-8')).get('metrics', {}).get('by_target', {}).get('chla', {}).get('h6', {})
                cov = chla.get('coverage_90')
            print(f'  {model} seed{s}: CRPS={crps} ΔCRPS={dcrps} cov90={cov}', flush=True)
            if crps is not None:
                vals.append(crps)
            if dcrps is not None:
                deltas.append(dcrps)
        if vals:
            mu = sum(vals) / len(vals)
            sd = (sum(((x - mu) ** 2 for x in vals)) / max(len(vals) - 1, 1)) ** 0.5
            print(f'  {model} CRPS mean±std = {mu:.4f} ± {sd:.4f}', flush=True)
        if deltas:
            mu = sum(deltas) / len(deltas)
            print(f'  {model} ΔCRPS mean = {mu:.4f}', flush=True)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--proc-dir', type=Path, default=ROOT / 'data/processed/lakebed_BVR_4h')
    ap.add_argument('--results-root', type=Path, default=ROOT / 'results/BVR_4h/phaseB')
    ap.add_argument('--seeds', default='0,1,2,3,4,5,6,7,8,9')
    ap.add_argument('--lab-seeds', default='0,1,2,3,4')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--block', default='all', choices=['all', 'crps', 'lab_thin', 'probe_loo', 'atlas'], help='Which Phase B block to run')
    ap.add_argument('--backbone', default=MAIN_BACKBONE, choices=['ionet_a'], help='main model (default: ionet_a)')
    ap.add_argument('--skip', default='', help='comma: main,lstm,qlstm,grud,persistence')
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    lab_seeds = [int(s) for s in args.lab_seeds.split(',') if s.strip()]
    skip = {x.strip() for x in args.skip.split(',') if x.strip()}
    proc = args.proc_dir
    root = args.results_root
    crps_root = root / 'crps'
    crps_root.mkdir(parents=True, exist_ok=True)
    backbone = args.backbone
    main_tag = resolve_backbone(backbone).result_tag
    if args.block in ('all', 'crps'):
        if 'persistence' not in skip:
            for s in seeds:
                eval_persistence(proc, crps_root / f'persistence_seed{s}', s, args.force)
        if 'main' not in skip:
            for s in seeds:
                out = crps_root / f'{main_tag}_seed{s}'
                train_eval_ionet_a(proc, out, s, args.device, args.force)
        if 'lstm' not in skip:
            for s in seeds:
                train_eval_baseline('lstm', proc, crps_root / f'lstm_seed{s}', s, args.device, args.force)
        if 'qlstm' not in skip:
            for s in seeds:
                train_eval_baseline('qlstm', proc, crps_root / f'qlstm_seed{s}', s, args.device, args.force)
        if 'grud' not in skip:
            for s in seeds:
                train_eval_baseline('grud', proc, crps_root / f'grud_seed{s}', s, args.device, args.force)
        summarize_crps(crps_root, [main_tag, 'persistence', 'lstm', 'qlstm', 'grud'], seeds)
    if args.block in ('all', 'lab_thin'):
        run([PY, str(ROOT / 'scripts/run_lab_thinning.py'), '--proc-dir', str(proc), '--out-dir', str(root / 'lab_thin'), '--seeds', ','.join((str(s) for s in lab_seeds)), '--backbone', backbone, '--device', args.device] + (['--force'] if args.force else []))
    if args.block in ('all', 'probe_loo'):
        run([PY, str(ROOT / 'scripts/run_probe_loo.py'), '--proc-dir', str(proc), '--out-dir', str(root / 'probe_loo'), '--seeds', ','.join((str(s) for s in lab_seeds)), '--backbone', backbone, '--device', args.device] + (['--force'] if args.force else []))
    if args.block in ('all', 'atlas'):
        run([PY, str(ROOT / 'scripts/plot_missingness_atlas.py'), '--proc-glob', 'data/processed/lakebed_*_4h', '--lakes', 'ME,BVR,FCR,TR,SP,BARC,CB,CRAM,TB,GL4,LIRO,PRLA,SUGG', '--out-json', str(root / 'missingness_atlas.json'), '--out-md', str(root / 'missingness_atlas.md')])
    print(f'\n[phaseB] done. Gate B: {main_tag} 24h ΔCRPS vs persistence > 0 (mean over seeds).', flush=True)
if __name__ == '__main__':
    main()
