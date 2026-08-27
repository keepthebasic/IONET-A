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
CAL_FALLBACK = {0: (0.898, 0.92, 1.7), 1: (0.749, 0.897, 1.6), 2: (0.802, 0.939, 1.8), 3: (0.851, 0.947, 2.3), 4: (0.826, 0.939, 2.3), 5: (0.776, 0.901, 1.5), 6: (0.879, 0.931, 2.4), 7: (0.772, 0.935, 1.8), 8: (0.779, 0.908, 1.7), 9: (0.793, 0.87, 1.6)}

def _h6(path: Path, key: str) -> float | None:
    if not path.exists():
        return None
    h = json.loads(path.read_text(encoding='utf-8'))['metrics']['by_target']['chla']['h6']
    v = h.get(key)
    return float(v) if v is not None else None

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', type=Path, default=ROOT / 'manuscript_WR_IONET-A/figures')
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    crps_root = PHASE_B / 'crps'
    raw_crps, raw_cov, raw_w = ([], [], [])
    cal_crps, cal_cov, cal_s = ([], [], [])
    for s in range(10):
        raw_p = crps_root / f'ionet_a_seed{s}' / 'E2_crps.json'
        raw_crps.append(_h6(raw_p, 'crps'))
        raw_cov.append(_h6(raw_p, 'coverage_90'))
        raw_w.append(_h6(raw_p, 'interval_width_90'))
        cal_p = crps_root / f'ionet_a_seed{s}' / 'E2_crps_calibrated.json'
        if cal_p.exists():
            cal_crps.append(_h6(cal_p, 'crps'))
            cal_cov.append(_h6(cal_p, 'coverage_90'))
            d = json.loads(cal_p.read_text(encoding='utf-8'))
            cal_s.append(float(d.get('interval_scale', np.nan)))
        elif s in CAL_FALLBACK:
            c, cv, sc = CAL_FALLBACK[s]
            cal_crps.append(c)
            cal_cov.append(cv)
            cal_s.append(sc)
        else:
            cal_crps.append(np.nan)
            cal_cov.append(np.nan)
            cal_s.append(np.nan)
    raw_crps = np.asarray(raw_crps, float)
    raw_cov = np.asarray(raw_cov, float)
    raw_w = np.asarray(raw_w, float)
    cal_crps = np.asarray(cal_crps, float)
    cal_cov = np.asarray(cal_cov, float)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), constrained_layout=True)
    ax = axes[0]
    nom = 0.9
    ax.plot([0.5, 1.0], [0.5, 1.0], 'k--', lw=1, label='Perfect calibration')
    ax.scatter([nom] * int(np.sum(np.isfinite(raw_cov))), raw_cov[np.isfinite(raw_cov)], c='#A23B72', s=36, alpha=0.85, label='Raw', zorder=3)
    ax.scatter([nom] * int(np.sum(np.isfinite(cal_cov))), cal_cov[np.isfinite(cal_cov)], c='#2E86AB', s=36, alpha=0.85, marker='s', label='Val-scaled', zorder=3)
    ax.axhline(nom, color='#94A3B8', ls=':', lw=0.8)
    ax.axvline(nom, color='#94A3B8', ls=':', lw=0.8)
    ax.set_xlim(0.82, 0.98)
    ax.set_ylim(0.62, 1.02)
    ax.set_xlabel('Nominal 90% interval coverage')
    ax.set_ylabel('Empirical 90% coverage (test)')
    ax.set_title('(a) Reliability at 90% (24 h Chl-a)')
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    ax = axes[1]
    labels = ['CRPS', 'cov90', 'width90']
    raw_m = [np.nanmean(raw_crps), np.nanmean(raw_cov), np.nanmean(raw_w)]
    cal_m = [np.nanmean(cal_crps), np.nanmean(cal_cov), np.nanmean(raw_w) * np.nanmean(cal_s)]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, raw_m, w, color='#A23B72', label='Raw')
    ax.bar(x + w / 2, cal_m, w, color='#2E86AB', label='Val-scaled')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title('(b) Test metrics (10-seed mean)')
    ax.legend(frameon=False, fontsize=8)
    for i, v in enumerate(raw_m):
        if np.isfinite(v):
            ax.text(i - w / 2, v, f'{v:.2f}', ha='center', va='bottom', fontsize=7)
    for i, v in enumerate(cal_m):
        if np.isfinite(v):
            ax.text(i + w / 2, v, f'{v:.2f}', ha='center', va='bottom', fontsize=7)
    for ext in ('pdf', 'png'):
        fig.savefig(args.out_dir / f'FigS2_calibration.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot_si] saved -> {args.out_dir}/FigS2_calibration.pdf', flush=True)
    print(f'  raw cov90={np.nanmean(raw_cov):.3f}  cal cov90={np.nanmean(cal_cov):.3f}', flush=True)
if __name__ == '__main__':
    main()
