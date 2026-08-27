from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--probe-dir', type=Path, default=ROOT / 'results/BVR_4h/phaseB/probe_loo')
    ap.add_argument('--seeds', default='0,1,2,3,4')
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    channels = ['temp', 'turbidity', 'do', 'ec']
    out: dict = {}
    n_mask = 0
    for ch in channels:
        vals = []
        for s in seeds:
            p = args.probe_dir / f'seed{s}' / f'E2_crps_loo_{ch}_mask.json'
            if not p.exists():
                continue
            n_mask += 1
            d = json.loads(p.read_text(encoding='utf-8'))
            vals.append(float(d['metrics']['by_target']['chla']['h6']['crps']))
        out[ch] = {'crps': vals}
    nat = []
    for s in seeds:
        p = args.probe_dir / f'seed{s}' / 'E2_crps_native.json'
        if p.exists():
            d = json.loads(p.read_text(encoding='utf-8'))
            nat.append(float(d['metrics']['by_target']['chla']['h6']['crps']))
    out['_native'] = {'crps': nat}
    dest = args.probe_dir / 'summary_mask.json'
    dest.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f'[export] mask files={n_mask} -> {dest}', flush=True)
    for ch in channels:
        v = out[ch]['crps']
        if v:
            m = sum(v) / len(v)
            print(f'  {ch}: mean={m:.4f} n={len(v)}', flush=True)
if __name__ == '__main__':
    main()
