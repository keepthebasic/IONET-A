from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PHASE_B = ROOT / 'results/BVR_4h/phaseB'
if not PHASE_B.exists():
    PHASE_B = ROOT / 'results/BVR_4h/phaseB'

def _crps(path: Path) -> float | None:
    if not path.exists():
        return None
    h = json.loads(path.read_text(encoding='utf-8'))['metrics']['by_target']['chla']['h6']
    v = h.get('crps')
    return float(v) if v is not None else None

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', type=Path, default=ROOT / 'manuscript_WR_IONET-A/figures')
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ratios = [('ratio_1p00', '100%'), ('ratio_0p50', '50%'), ('ratio_0p25', '25%'), ('ratio_0p10', '10%')]
    lab_root = PHASE_B / 'lab_thin'
    x = np.arange(len(ratios))
    colors = ['#A23B72', '#2E86AB', '#E63946', '#457B9D']
    fig, ax = plt.subplots(figsize=(5.5, 3.8), constrained_layout=True)
    for i, (tag, label) in enumerate(ratios):
        vals = []
        for s in range(5):
            v = _crps(lab_root / tag / f'seed{s}' / 'E2_crps.json')
            if v is not None:
                vals.append(v)
                ax.scatter(i + (s - 2) * 0.06, v, color=colors[i], s=28, alpha=0.85, zorder=3)
        if vals:
            m = float(np.mean(vals))
            ax.hlines(m, i - 0.22, i + 0.22, colors=colors[i], lw=2, zorder=2)
            ax.text(i + 0.24, m, f'{m:.3f}', fontsize=7, va='center')
    ax.set_xticks(x)
    ax.set_xticklabels([r[1] for r in ratios])
    ax.set_xlabel('Training assay retention')
    ax.set_ylabel('24 h Chl-a CRPS')
    ax.set_title('Lab thinning: individual seeds (dots) and mean (line)')
    for ext in ('pdf', 'png'):
        fig.savefig(args.out_dir / f'FigS1_lab_thin_seeds.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot_si] saved -> {args.out_dir}/FigS1_lab_thin_seeds.pdf')
if __name__ == '__main__':
    main()
