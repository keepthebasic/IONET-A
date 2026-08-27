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
PHASE_A = ROOT / 'results/BVR_4h/phaseA'
PHASE_B = ROOT / 'results/BVR_4h/phaseB'
LOLO = ROOT / 'results/BVR_4h/lolo'
if not PHASE_B.exists():
    PHASE_B = ROOT / 'results/BVR_4h/phaseB'

def _set_style() -> None:
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'], 'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.linewidth': 0.8, 'xtick.direction': 'out', 'ytick.direction': 'out', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300, 'figure.facecolor': 'white', 'axes.facecolor': 'white', 'axes.spines.top': False, 'axes.spines.right': False})

def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))

def _metric_h6(path: Path, target: str, key: str) -> float | None:
    if not path.exists():
        return None
    d = _load(path)
    h = d.get('metrics', {}).get('by_target', {}).get(target, {}).get('h6', {})
    v = h.get(key)
    return float(v) if v is not None else None

def _collect_seeds(root: Path, pattern: str, json_name: str, target: str, key: str, seeds: range | list[int]) -> np.ndarray:
    vals = []
    for s in seeds:
        p = root / pattern.format(s=s) / json_name
        v = _metric_h6(p, target, key)
        if v is not None:
            vals.append(v)
    return np.asarray(vals, dtype=float)

def _mean_std(vals: np.ndarray) -> tuple[float, float]:
    if vals.size == 0:
        return (float('nan'), float('nan'))
    if vals.size == 1:
        return (float(vals[0]), 0.0)
    return (float(vals.mean()), float(vals.std(ddof=1)))

def _save(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(out.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[plot] saved -> {out.with_suffix('.pdf')}  (+ .png)")

def plot_fig2(out_dir: Path) -> None:
    atlas_path = PHASE_B / 'missingness_atlas.json'
    if not atlas_path.exists():
        print(f'[plot] skip Fig.2: missing {atlas_path}')
        return
    atlas = _load(atlas_path)
    lakes = atlas['lakes']
    ids = [r['lake_id'] for r in lakes]
    chla = [100 * float(r['missing_rate_soft'].get('chla', float('nan'))) for r in lakes]
    do = [100 * float(r['missing_rate_easy'].get('do', float('nan'))) for r in lakes]
    turb = [100 * float(r['missing_rate_easy'].get('turbidity', float('nan'))) for r in lakes]
    gap_p90 = [float(r.get('gap_len_p90_steps', float('nan'))) for r in lakes]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    x = np.arange(len(ids))
    w = 0.28
    ax = axes[0]
    b1 = ax.bar(x - w, chla, w, color='#A23B72')
    b2 = ax.bar(x, do, w, color='#2E86AB')
    b3 = ax.bar(x + w, turb, w, color='#48CAE4')
    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Native missingness (%)')
    ax.set_ylim(0, 105)
    ax.axhline(90, color='#94A3B8', ls='--', lw=0.8, zorder=0)
    ax.set_title('(a) Missingness by lake', pad=8)
    ax = axes[1]
    ax.bar(x, gap_p90, color='#457B9D', width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Gap length p90 (4 h steps)')
    ymax = max(gap_p90) if gap_p90 else 1.0
    ax.set_ylim(0, ymax * 1.12)
    ax.set_title('(b) Probe dropout duration', pad=8)
    fig.legend(handles=[b1, b2, b3], labels=['Chl-a (lab)', 'DO (probe)', 'Turbidity'], loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.92), columnspacing=1.4, handlelength=1.3)
    fig.suptitle(f"Incomplete monitoring across {atlas['n_lakes']} LakeBeD lakes (4 h product)", fontsize=10, y=0.98)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.86))
    _save(fig, out_dir / 'Fig2_missingness_atlas')

def plot_fig3(out_dir: Path) -> None:
    seeds = range(10)
    lstm_n = _collect_seeds(PHASE_A / 'native', 'lstm_seed{s}', 'E2_metrics.json', 'chla', 'nse', seeds)
    lstm_i = _collect_seeds(PHASE_A / 'impute', 'lstm_seed{s}', 'E2_metrics.json', 'chla', 'nse', seeds)
    lstm_l = _collect_seeds(PHASE_A / 'locf', 'lstm_seed{s}', 'E2_metrics.json', 'chla', 'nse', seeds)
    grud_n = _collect_seeds(PHASE_A / 'native', 'grud_seed{s}', 'E2_metrics.json', 'chla', 'nse', seeds)
    crps_root = PHASE_B / 'crps'
    persist = _collect_seeds(crps_root, 'persistence_seed{s}', 'E2_crps.json', 'chla', 'crps', seeds)
    lstm_c = _collect_seeds(crps_root, 'lstm_seed{s}', 'E2_crps.json', 'chla', 'crps', seeds)
    qlstm_c = _collect_seeds(crps_root, 'qlstm_seed{s}', 'E2_crps.json', 'chla', 'crps', seeds)
    grud_c = _collect_seeds(crps_root, 'grud_seed{s}', 'E2_crps.json', 'chla', 'crps', seeds)
    ionet = _collect_seeds(crps_root, 'ionet_a_seed{s}', 'E2_crps.json', 'chla', 'crps', seeds)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), constrained_layout=True)

    def _annotate_bars(ax, bars, means, stds, fmt: str='{:.3f}', pad_frac: float=0.05) -> None:
        y0, y1 = ax.get_ylim()
        span = max(y1 - y0, 1e-06)
        pad = pad_frac * span
        halo = dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.98)
        for b, m, s in zip(bars, means, stds):
            if not np.isfinite(m):
                continue
            s = float(s) if np.isfinite(s) else 0.0
            x = b.get_x() + b.get_width() / 2 + 0.14
            if m >= 0:
                y = m + s + pad
                ax.text(x, y, fmt.format(m), ha='left', va='bottom', fontsize=7, bbox=halo, zorder=5, clip_on=False)
            else:
                y = m - s - pad
                ax.text(x, y, fmt.format(m), ha='left', va='top', fontsize=7, bbox=halo, zorder=5, clip_on=False)
    ax = axes[0]
    labels = ['LSTM\nnative', 'LSTM\nlinear', 'LSTM\nLOCF', 'GRU-D\nnative']
    means = [_mean_std(v)[0] for v in (lstm_n, lstm_i, lstm_l, grud_n)]
    stds = [_mean_std(v)[1] for v in (lstm_n, lstm_i, lstm_l, grud_n)]
    colors = ['#2E86AB', '#E63946', '#F4A261', '#457B9D']
    bars = ax.bar(np.arange(4), means, yerr=stds, color=colors, capsize=2, width=0.62, ecolor='#334155')
    ax.axhline(0, color='#94A3B8', lw=0.8)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(labels)
    ax.set_ylabel('24 h Chl-a NSE')
    ax.set_title('(a) Impute-then-forecast is harmful')
    y_hi = max((m + s for m, s in zip(means, stds) if np.isfinite(m)), default=0.0)
    y_lo = min((m - s for m, s in zip(means, stds) if np.isfinite(m)), default=0.0)
    pad = 0.16 * max(y_hi - y_lo, 0.2)
    ax.set_ylim(y_lo - pad, y_hi + pad)
    ax.set_xlim(-0.6, 3.7)
    _annotate_bars(ax, bars, means, stds, fmt='{:.3f}', pad_frac=0.045)
    ax = axes[1]
    labels2 = ['Persistence', 'LSTM', 'Q-LSTM', 'GRU-D', 'IO-Net-A']
    series = [persist, lstm_c, qlstm_c, grud_c, ionet]
    means2 = [_mean_std(v)[0] for v in series]
    stds2 = [_mean_std(v)[1] for v in series]
    colors2 = ['#94A3B8', '#2E86AB', '#1D3557', '#457B9D', '#A23B72']
    bars2 = ax.bar(np.arange(5), means2, yerr=stds2, color=colors2, capsize=2, width=0.65, ecolor='#334155')
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(labels2, rotation=20, ha='right')
    ax.set_ylabel('24 h Chl-a CRPS (lower better)')
    ax.set_title('(b) Probabilistic skill vs baselines')
    finite = [(m, s) for m, s in zip(means2, stds2) if np.isfinite(m)]
    y_hi2 = max((m + s for m, s in finite), default=1.0) if finite else 1.0
    ax.set_ylim(0, y_hi2 * 1.22)
    ax.set_xlim(-0.6, 4.7)
    _annotate_bars(ax, bars2, means2, stds2, fmt='{:.2f}', pad_frac=0.035)
    _save(fig, out_dir / 'Fig3_impute_harm_and_baselines')

def plot_fig4(out_dir: Path) -> None:
    seeds = range(10)
    crps_root = PHASE_B / 'crps'
    models = [('persistence', 'Persistence', '#94A3B8'), ('lstm', 'LSTM', '#2E86AB'), ('qlstm', 'Q-LSTM', '#1D3557'), ('grud', 'GRU-D', '#457B9D'), ('ionet_a', 'IO-Net-A', '#A23B72')]
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.5), constrained_layout=True)
    ax = axes[0]
    xs, means, stds, cols = ([], [], [], [])
    for tag, name, col in models:
        vals = _collect_seeds(crps_root, f'{tag}_seed{{s}}', 'E2_crps.json', 'chla', 'crps', seeds)
        m, s = _mean_std(vals)
        xs.append(name)
        means.append(m)
        stds.append(s)
        cols.append(col)
    ax.bar(np.arange(len(xs)), means, yerr=stds, color=cols, capsize=3, ecolor='#334155')
    ax.set_xticks(np.arange(len(xs)))
    ax.set_xticklabels(xs, rotation=15, ha='right')
    ax.set_ylabel('24 h Chl-a CRPS')
    ax.set_title('(a) CRPS')
    ax = axes[1]
    dvals = _collect_seeds(crps_root, 'ionet_a_seed{s}', 'E2_crps.json', 'chla', 'delta_crps_vs_persistence', seeds)
    ax.bar(np.arange(len(dvals)), dvals, color='#A23B72', width=0.7)
    ax.axhline(0, color='#94A3B8', lw=0.8)
    mu, sd = _mean_std(dvals)
    ax.axhline(mu, color='#1D3557', ls='--', lw=1.0)
    ax.set_xlabel('Seed')
    ax.set_ylabel('ΔCRPS vs persistence')
    ax.set_title(f'(b) IO-Net-A skill gain\nmean = {mu:.3f} ± {sd:.3f}', fontsize=9)
    ax = axes[2]
    cov = _collect_seeds(crps_root, 'ionet_a_seed{s}', 'E2_crps.json', 'chla', 'coverage_90', seeds)
    ax.bar(np.arange(len(cov)), cov, color='#48CAE4', width=0.7)
    ax.axhline(0.9, color='#E63946', ls='--', lw=1.0)
    cmu, csd = _mean_std(cov)
    ax.axhline(cmu, color='#1D3557', ls=':', lw=1.0)
    ax.set_xlabel('Seed')
    ax.set_ylabel('90% interval coverage')
    ax.set_ylim(0, 1.05)
    ax.set_title(f'(c) Calibration (no post-hoc)\nnominal 90%; mean = {cmu:.2f}', fontsize=9)
    _save(fig, out_dir / 'Fig4_crps_main')

def plot_fig5(out_dir: Path) -> None:
    lab_root = PHASE_B / 'lab_thin'
    ratios = [('ratio_1p00', '100%'), ('ratio_0p50', '50%'), ('ratio_0p25', '25%'), ('ratio_0p10', '10%')]
    lab_means, lab_stds = ([], [])
    for tag, _ in ratios:
        vals = []
        for s in range(5):
            p = lab_root / tag / f'seed{s}' / 'E2_crps.json'
            v = _metric_h6(p, 'chla', 'crps')
            if v is not None:
                vals.append(v)
        m, sd = _mean_std(np.asarray(vals, dtype=float))
        lab_means.append(m)
        lab_stds.append(sd)
    loo_path = PHASE_B / 'probe_loo' / 'summary.json'
    mask_summary_path = PHASE_B / 'probe_loo' / 'summary_mask.json'
    channels = ['temp', 'turbidity', 'do', 'ec']
    native_vals: list[float] = []
    for s in range(5):
        p = PHASE_B / 'probe_loo' / f'seed{s}' / 'E2_crps_native.json'
        v = _metric_h6(p, 'chla', 'crps')
        if v is not None:
            native_vals.append(v)
    if mask_summary_path.exists():
        sm_nat = [float(x) for x in _load(mask_summary_path).get('_native', {}).get('crps', []) if np.isfinite(x)]
        if len(sm_nat) >= 3:
            native_vals = sm_nat
    native_mean = float(np.mean(native_vals)) if native_vals else float('nan')
    loo_m, loo_s = ([], [])
    used_mask = False
    for ch in channels:
        vals = []
        for s in range(5):
            p = PHASE_B / 'probe_loo' / f'seed{s}' / f'E2_crps_loo_{ch}_mask.json'
            v = _metric_h6(p, 'chla', 'crps')
            if v is not None:
                vals.append(v)
        if vals:
            used_mask = True
        elif mask_summary_path.exists():
            vals = [float(x) for x in _load(mask_summary_path).get(ch, {}).get('crps', []) if np.isfinite(x)]
            if vals:
                used_mask = True
        m, sd = _mean_std(np.asarray(vals, dtype=float))
        loo_m.append(m)
        loo_s.append(sd)
    if not used_mask:
        print(f'[plot] Fig.5 right: no mask LOO data — sync seed*/E2_crps_loo_*_mask.json or probe_loo/summary_mask.json (legacy zero JSON ignored)', flush=True)
        channels = []
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), constrained_layout=True)
    ax = axes[0]
    x = np.arange(len(ratios))
    ax.errorbar(x, lab_means, yerr=lab_stds, fmt='-o', color='#A23B72', capsize=3, lw=1.5, markersize=6)
    persist = _collect_seeds(PHASE_B / 'crps', 'persistence_seed{s}', 'E2_crps.json', 'chla', 'crps', range(10))
    if persist.size:
        ax.axhline(float(persist.mean()), color='#94A3B8', ls='--', lw=1.0, label='Persistence')
    ax.set_xticks(x)
    ax.set_xticklabels([r[1] for r in ratios])
    ax.set_xlabel('Training assay retention')
    ax.set_ylabel('24 h Chl-a CRPS')
    ax.set_title('(a) Lab assay thinning')
    ax.legend(frameon=False)
    ax = axes[1]
    if channels:
        labels = ['native', '−temp', '−turbidity', '−DO', '−EC']
        means = [native_mean] + loo_m
        stds = [0.0] + loo_s
        if native_vals:
            stds[0] = _mean_std(np.asarray(native_vals))[1]
        cols = ['#1D3557', '#2E86AB', '#E63946', '#457B9D', '#94A3B8']
        ax.bar(np.arange(len(labels)), means, yerr=stds, color=cols, capsize=3, ecolor='#334155')
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha='right')
        ax.set_ylabel('24 h Chl-a CRPS')
        ax.set_title('(b) Probe LOO (mask dropout)', fontsize=9)
        tip = max((m + (s if np.isfinite(s) else 0.0) for m, s in zip(means, stds)), default=1.0)
        ax.set_ylim(0, tip * 1.12)
    _save(fig, out_dir / 'Fig5_lab_thin_probe_loo')

def plot_fig6(out_dir: Path) -> None:
    summary_path = LOLO / 'lolo_summary.json'
    if not summary_path.exists():
        print(f'[plot] skip Fig.6: missing {summary_path}')
        return
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    folds = summary.get('folds', {})
    lake_order = summary.get('lakes') or list(folds.keys())

    def _seed_vals(fold: dict, key: str) -> np.ndarray:
        vals = []
        for seed_blob in fold.get('seeds', {}).values():
            v = seed_blob.get(key)
            if v is None:
                continue
            fv = float(v)
            if np.isfinite(fv):
                vals.append(fv)
        return np.asarray(vals, dtype=float)
    inlake = _collect_seeds(PHASE_B / 'crps', 'ionet_a_seed{s}', 'E2_crps.json', 'chla', 'crps', range(10))
    inlake_mu, _ = _mean_std(inlake)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5), constrained_layout=True)
    ax = axes[0]
    reportable = []
    for lk in lake_order:
        fold = folds.get(lk, {})
        vals = _seed_vals(fold, 'chla_crps')
        if vals.size:
            reportable.append((lk, vals))
    if reportable:
        xs = np.arange(len(reportable))
        means = [float(v.mean()) for _, v in reportable]
        stds = [float(v.std(ddof=1)) if v.size > 1 else 0.0 for _, v in reportable]
        cols = ['#A23B72' if lk == 'BVR' else '#E63946' for lk, _ in reportable]
        ax.bar(xs, means, yerr=stds, color=cols, capsize=3, ecolor='#334155', width=0.55)
        ax.set_xticks(xs)
        ax.set_xticklabels([lk for lk, _ in reportable])
        if np.isfinite(inlake_mu):
            ax.axhline(inlake_mu, color='#1D3557', ls='--', lw=1.0)
        ax.set_ylabel('24 h Chl-a CRPS (lower better)')
        nr = [lk for lk in lake_order if not _seed_vals(folds.get(lk, {}), 'chla_crps').size]
        line2_bits = []
        if nr:
            line2_bits.append(f"{', '.join(nr)} NR (n_valid<30)")
        if np.isfinite(inlake_mu):
            line2_bits.append(f'dashed: BVR in-lake={inlake_mu:.2f}')
        if line2_bits:
            ax.set_title('(a) Leave-one-lake-out Chl-a\n' + '; '.join(line2_bits), fontsize=8)
        else:
            ax.set_title('(a) Leave-one-lake-out Chl-a', fontsize=9)
        ymax = max((m + s for m, s in zip(means, stds)))
        ax.set_ylim(0, ymax * 1.12)
    ax = axes[1]
    do_means, do_stds, do_labs = ([], [], [])
    for lk in lake_order:
        vals = _seed_vals(folds.get(lk, {}), 'do_delta_crps')
        if not vals.size:
            continue
        do_labs.append(lk)
        do_means.append(float(vals.mean()))
        do_stds.append(float(vals.std(ddof=1)) if vals.size > 1 else 0.0)
    xs = np.arange(len(do_labs))
    colors = ['#2E86AB' if m >= 0 else '#E63946' for m in do_means]
    ax.bar(xs, do_means, yerr=do_stds, color=colors, capsize=3, ecolor='#334155', width=0.65)
    ax.axhline(0, color='#94A3B8', lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(do_labs)
    ax.set_ylabel('24 h DO ΔCRPS vs persistence')
    ax.set_title('(b) DO transfer (all folds)', fontsize=9)
    hi = max((m + s for m, s in zip(do_means, do_stds)))
    if min(do_means) >= 0:
        ax.set_ylim(0, hi * 1.15)
    else:
        lo = min((m - s for m, s in zip(do_means, do_stds)))
        pad = 0.12 * max(hi - lo, 0.2)
        ax.set_ylim(lo - pad, hi + pad)
    fig.suptitle('Cross-lake transfer boundary (IO-Net-A, 5 seeds)', fontsize=10, y=1.04)
    _save(fig, out_dir / 'Fig6_lolo_transfer_boundary')

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', type=Path, default=ROOT / 'manuscript_WR_IONET-A/figures')
    ap.add_argument('--fig', default='all', choices=['all', 'fig2', 'fig3', 'fig4', 'fig5', 'fig6'])
    args = ap.parse_args()
    _set_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[plot] PHASE_A={PHASE_A.exists()}  PHASE_B={PHASE_B}  LOLO={LOLO.exists()}', flush=True)
    if args.fig in ('all', 'fig2'):
        plot_fig2(args.out_dir)
    if args.fig in ('all', 'fig3'):
        plot_fig3(args.out_dir)
    if args.fig in ('all', 'fig4'):
        plot_fig4(args.out_dir)
    if args.fig in ('all', 'fig5'):
        plot_fig5(args.out_dir)
    if args.fig in ('all', 'fig6'):
        plot_fig6(args.out_dir)
    print('[plot] done.', flush=True)
if __name__ == '__main__':
    main()
