from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.backbone import resolve_backbone

def run(cmd: list[str]) -> None:
    print('+', ' '.join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if proc.returncode != 0:
        raise SystemExit(f"failed ({proc.returncode}): {' '.join(cmd)}")

def _metric(path: Path, key: str='crps') -> float | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding='utf-8'))
    met = d.get('metrics', d)
    chla = met.get('by_target', {}).get('chla', {}).get('h6', {})
    v = chla.get(key)
    return float(v) if v is not None else None

def _ckpt_is_ionet_a(ckpt_path: Path) -> bool:
    if not ckpt_path.exists():
        return False
    try:
        import torch
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    except Exception:
        return False
    if ckpt.get('model') not in ('ionet_a', 'ionet_lite'):
        return False
    return bool(ckpt.get('ionet_a_cfg') or ckpt.get('ionet_lite_cfg'))

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--proc-dir', type=Path, default=ROOT / 'data/processed/lakebed_BVR_4h')
    ap.add_argument('--out-dir', type=Path, default=ROOT / 'results/BVR_4h/phaseB/lab_thin')
    ap.add_argument('--ratios', default='1.0,0.5,0.25,0.1')
    ap.add_argument('--seeds', default='0,1,2,3,4')
    ap.add_argument('--backbone', default='ionet_a', choices=['ionet_a'])
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    ratios = [float(r) for r in args.ratios.split(',') if r.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, list[float]] = {}
    for ratio in ratios:
        tag = f'ratio_{ratio:.2f}'.replace('.', 'p')
        summary[tag] = []
        for seed in seeds:
            run_root = args.out_dir / tag / f'seed{seed}'
            if ratio >= 1.0:
                train_proc = args.proc_dir
            else:
                train_proc = args.out_dir / 'data' / tag / f'seed{seed}'
                if not (train_proc / 'train.npz').exists() or args.force:
                    run([PY, str(ROOT / 'scripts/lab_thin_data.py'), '--proc-dir', str(args.proc_dir), '--out-dir', str(train_proc), '--ratio', str(ratio), '--seed', str(seed)])
            ckpt = run_root / 'best.pt'
            crps_path = run_root / 'E2_crps.json'
            spec = resolve_backbone(args.backbone)
            train_script = spec.train_script
            eval_script = spec.eval_script
            hp = spec.hp_file
            reused_main = False
            if ratio >= 1.0 and spec.result_tag == 'ionet_a':
                main_ckpt = args.out_dir.parent / 'crps' / f'ionet_a_seed{seed}' / 'best.pt'
                if main_ckpt.exists() and _ckpt_is_ionet_a(main_ckpt) and (not args.force):
                    import shutil
                    run_root.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(main_ckpt, ckpt)
                    reused_main = True
                    print(f'[lab_thin] ratio=1.0 reuse {main_ckpt} -> {ckpt}', flush=True)
                elif main_ckpt.exists() and (not _ckpt_is_ionet_a(main_ckpt)):
                    print(f'[lab_thin] WARN: skip invalid main ckpt {main_ckpt}', flush=True)
            if spec.result_tag == 'ionet_a' and ckpt.exists() and (not _ckpt_is_ionet_a(ckpt)):
                print(f'[lab_thin] remove stale non-ionet_a ckpt -> {ckpt}', flush=True)
                ckpt.unlink(missing_ok=True)
            need_train = not reused_main and (args.force or not _ckpt_is_ionet_a(ckpt))
            need_eval = args.force or not crps_path.exists()
            if need_train:
                run([PY, str(ROOT / f'scripts/{train_script}'), '--proc-dir', str(train_proc), '--hyperparams', str(ROOT / f'configs/{hp}'), '--seed', str(seed), '--device', args.device, '--out-dir', str(run_root), '--force'])
            if need_eval or need_train:
                run([PY, str(ROOT / f'scripts/{eval_script}'), '--proc-dir', str(args.proc_dir), '--results-dir', str(run_root), '--out', str(crps_path), '--denormalize', '1', '--calibrate', '0', '--device', args.device])
            crps = _metric(crps_path, 'crps')
            dcrps = _metric(crps_path, 'delta_crps_vs_persistence')
            print(f'[lab_thin] ratio={ratio} seed={seed} CRPS={crps} ΔCRPS={dcrps}', flush=True)
            if crps is not None:
                summary[tag].append(crps)
    print('\n=== Lab thinning summary (24h Chl-a CRPS) ===', flush=True)
    for tag, vals in summary.items():
        if vals:
            mu = sum(vals) / len(vals)
            sd = (sum(((x - mu) ** 2 for x in vals)) / max(len(vals) - 1, 1)) ** 0.5
            print(f'  {tag}: {mu:.4f} ± {sd:.4f}  n={len(vals)}', flush=True)
    (args.out_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
if __name__ == '__main__':
    main()
