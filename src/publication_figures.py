"""
Publication-Quality Figures for AMRI Paper
============================================
Generates 8 figures for the main paper and supplement.

Main paper:
  Fig 1: Coverage vs delta for all methods (the money plot)
  Fig 2: AMRI v2 vs AKS head-to-head comparison
  Fig 3: Real data dashboard (61 datasets)
  Fig 4: Blending weight behavior
  Fig 5: Efficiency under correct specification

Supplement:
  Fig S1: Per-DGP coverage breakdown
  Fig S2: Width comparison across methods
  Fig S3: Coverage by sample size convergence

Usage:
  python src/publication_figures.py           # Generate all from saved CSVs
  python src/publication_figures.py --quick   # Quick mode with reduced sims
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Style setup for publication
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Color scheme
COLORS = {
    'AMRI_v2': '#2166AC',      # strong blue
    'AMRI_v1': '#67A9CF',      # light blue
    'Sandwich_HC3': '#D6604D', # red
    'Sandwich_HC4': '#F4A582', # light red
    'AKS_Adaptive': '#5AB4AC', # teal
    'Naive_OLS': '#999999',    # gray
    'Pairs_Bootstrap': '#B2ABD2',
    'Wild_Bootstrap': '#FDB863',
    'Bootstrap_t': '#E08214',
}


def load_competitor_data():
    """Load competitor comparison results."""
    f = RESULTS_DIR / "results_competitor_comparison.csv"
    if f.exists():
        return pd.read_csv(f)
    return None


def load_real_data():
    """Load real data validation results."""
    f = FIGURES_DIR / "real_data_results.csv"
    if f.exists():
        return pd.read_csv(f)
    return None


def load_theorem_data():
    """Load formal theory verification results."""
    dfs = {}
    for i, name in [(1, 'continuity'), (2, 'coverage'), (3, 'efficiency')]:
        f = RESULTS_DIR / f"results_theorem{i}_{name}.csv"
        if f.exists():
            dfs[f'theorem{i}'] = pd.read_csv(f)
    return dfs


# ============================================================================
# FIGURE 1: Coverage vs Delta (Money Plot)
# ============================================================================
def fig_coverage_vs_delta(df):
    """Main result figure: coverage as a function of misspecification severity."""
    if df is None:
        print("  No competitor data available, skipping Fig 1")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    dgps = sorted(df['dgp'].unique())[:3]
    methods_to_plot = ['Naive_OLS', 'Sandwich_HC3', 'AKS_Adaptive', 'AMRI_v2']

    for ax_idx, dgp in enumerate(dgps):
        ax = axes[ax_idx]
        dgp_data = df[df['dgp'] == dgp]

        for method in methods_to_plot:
            mdata = dgp_data[dgp_data['method'] == method]
            if len(mdata) == 0:
                continue

            # Average across sample sizes for each delta
            avg = mdata.groupby('delta')['coverage'].mean()
            label = method.replace('_', ' ')
            if method == 'AMRI_v2':
                label = 'AMRI v2 (ours)'

            ax.plot(avg.index, avg.values, 'o-',
                    color=COLORS.get(method, '#333333'),
                    label=label,
                    linewidth=2 if method == 'AMRI_v2' else 1.5,
                    markersize=6 if method == 'AMRI_v2' else 4,
                    zorder=5 if method == 'AMRI_v2' else 3)

        ax.axhline(y=0.95, color='black', linestyle='--', linewidth=0.8,
                   alpha=0.7, label='Nominal 95%')
        ax.axhspan(0.93, 0.97, alpha=0.08, color='green')
        ax.set_xlabel(r'Misspecification severity ($\delta$)')
        ax.set_title(dgp.replace('_', ' '))
        ax.set_ylim(0.70, 1.0)

    axes[0].set_ylabel('Coverage probability')
    axes[0].legend(loc='lower left', framealpha=0.9)

    fig.suptitle('Coverage Under Increasing Misspecification', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig1_coverage_vs_delta.png')
    fig.savefig(FIGURES_DIR / 'fig1_coverage_vs_delta.pdf')
    plt.close(fig)
    print("  Fig 1: Coverage vs Delta saved")


# ============================================================================
# FIGURE 2: AMRI v2 vs AKS Head-to-Head
# ============================================================================
def fig_amri_vs_aks(df):
    """Scatter plot comparing AMRI v2 and AKS coverage accuracy."""
    if df is None:
        print("  No competitor data available, skipping Fig 2")
        return

    amri = df[df['method'] == 'AMRI_v2'][['dgp', 'delta', 'n', 'coverage', 'avg_width']].copy()
    aks = df[df['method'] == 'AKS_Adaptive'][['dgp', 'delta', 'n', 'coverage', 'avg_width']].copy()

    merged = amri.merge(aks, on=['dgp', 'delta', 'n'], suffixes=('_amri', '_aks'))
    if len(merged) == 0:
        print("  No matched AMRI/AKS data, skipping Fig 2")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    # Panel A: Coverage accuracy scatter
    amri_acc = abs(merged['coverage_amri'] - 0.95)
    aks_acc = abs(merged['coverage_aks'] - 0.95)

    colors = merged['delta'].map({0.0: '#4DAF4A', 0.5: '#FF7F00', 1.0: '#E41A1C'})
    ax1.scatter(aks_acc, amri_acc, c=colors, s=60, alpha=0.8, edgecolors='white', linewidth=0.5)
    lim = max(aks_acc.max(), amri_acc.max()) * 1.1
    ax1.plot([0, lim], [0, lim], 'k--', linewidth=0.8, alpha=0.5, label='Equal accuracy')
    ax1.set_xlabel('AKS coverage accuracy ($|$cov $- 0.95|$)')
    ax1.set_ylabel('AMRI v2 coverage accuracy ($|$cov $- 0.95|$)')
    ax1.set_title('A. Coverage Accuracy Comparison')
    ax1.legend()
    ax1.text(0.05, 0.95, 'AMRI better\n(below line)',
             transform=ax1.transAxes, fontsize=9, va='top', style='italic', color='gray')

    # Count wins
    n_amri_better = (amri_acc < aks_acc).sum()
    n_total = len(merged)
    ax1.text(0.95, 0.05, f'AMRI wins: {n_amri_better}/{n_total}',
             transform=ax1.transAxes, fontsize=10, ha='right', fontweight='bold',
             color=COLORS['AMRI_v2'])

    # Panel B: Coverage distributions
    bins = np.linspace(0.88, 1.0, 25)
    ax2.hist(merged['coverage_amri'], bins=bins, alpha=0.7, color=COLORS['AMRI_v2'],
             label=f'AMRI v2 (mean={merged["coverage_amri"].mean():.3f})', edgecolor='white')
    ax2.hist(merged['coverage_aks'], bins=bins, alpha=0.5, color=COLORS['AKS_Adaptive'],
             label=f'AKS (mean={merged["coverage_aks"].mean():.3f})', edgecolor='white')
    ax2.axvline(0.95, color='black', linestyle='--', linewidth=0.8)
    ax2.set_xlabel('Coverage probability')
    ax2.set_ylabel('Count')
    ax2.set_title('B. Coverage Distribution')
    ax2.legend()

    fig.suptitle('AMRI v2 vs AKS Adaptive CI Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig2_amri_vs_aks.png')
    fig.savefig(FIGURES_DIR / 'fig2_amri_vs_aks.pdf')
    plt.close(fig)
    print("  Fig 2: AMRI vs AKS saved")


# ============================================================================
# FIGURE 3: Real Data Dashboard
# ============================================================================
def fig_real_data_dashboard(rdf):
    """Dashboard showing AMRI performance on 61 real datasets."""
    if rdf is None:
        print("  No real data results available, skipping Fig 3")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel A: SE ratio distribution
    ax = axes[0, 0]
    se_ratios = rdf['se_ratio'].dropna()
    ax.hist(se_ratios, bins=30, color=COLORS['AMRI_v2'], alpha=0.7, edgecolor='white')
    ax.axvline(1.0, color='red', linestyle='--', linewidth=1, label='No misspecification')
    ax.set_xlabel('SE ratio (HC3/Naive)')
    ax.set_ylabel('Count')
    ax.set_title(f'A. SE Ratio Distribution (n={len(rdf)} datasets)')
    ax.legend()
    n_misspec = (se_ratios > 1.1).sum()
    ax.text(0.95, 0.95, f'{n_misspec}/{len(rdf)} datasets\nshow misspecification',
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Panel B: Bootstrap coverage comparison
    ax = axes[0, 1]
    methods = ['naive_boot_coverage', 'hc3_boot_coverage', 'amri_v2_boot_coverage']
    labels = ['Naive', 'HC3', 'AMRI v2']
    colors_bp = [COLORS['Naive_OLS'], COLORS['Sandwich_HC3'], COLORS['AMRI_v2']]

    data = [rdf[m].dropna().values for m in methods]
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
    for patch, color in zip(bp['boxes'], colors_bp):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.axhline(0.95, color='black', linestyle='--', linewidth=0.8)
    ax.set_ylabel('Bootstrap coverage')
    ax.set_title('B. Coverage Across Real Datasets')

    # Panel C: SE error comparison
    ax = axes[1, 0]
    methods_err = ['naive_se_error_pct', 'hc3_se_error_pct', 'amri_v2_se_error_pct']
    labels_err = ['Naive', 'HC3', 'AMRI v2']
    data_err = [rdf[m].dropna().values for m in methods_err]
    bp2 = ax.boxplot(data_err, labels=labels_err, patch_artist=True, widths=0.5)
    for patch, color in zip(bp2['boxes'], colors_bp):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('SE Error (% from bootstrap truth)')
    ax.set_title('C. SE Estimation Accuracy')
    ax.set_ylim(0, min(100, np.percentile(np.concatenate(data_err), 95) * 1.3))

    # Panel D: AMRI v2 weight distribution
    ax = axes[1, 1]
    if 'amri_v2_weight' in rdf.columns:
        weights = rdf['amri_v2_weight'].dropna()
        ax.hist(weights, bins=20, color=COLORS['AMRI_v2'], alpha=0.7, edgecolor='white')
        ax.set_xlabel('AMRI v2 blending weight $w$')
        ax.set_ylabel('Count')
        ax.set_title('D. Adaptive Weight Distribution')
        n_zero = (weights == 0).sum()
        n_one = (weights == 1).sum()
        n_mid = len(weights) - n_zero - n_one
        ax.text(0.95, 0.95, f'w=0: {n_zero}, 0<w<1: {n_mid}, w=1: {n_one}',
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle(f'Real Data Validation: {len(rdf)} Datasets', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig3_real_data_dashboard.png')
    fig.savefig(FIGURES_DIR / 'fig3_real_data_dashboard.pdf')
    plt.close(fig)
    print("  Fig 3: Real Data Dashboard saved")


# ============================================================================
# FIGURE 4: Blending Weight Behavior
# ============================================================================
def fig_blending_weight():
    """Show how the blending weight w varies with SE ratio and sample size."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel A: w vs log(SE ratio) for different n
    ratios = np.linspace(0.5, 3.0, 200)
    for n, ls in [(50, ':'), (100, '--'), (500, '-'), (2000, '-')]:
        weights = []
        for r in ratios:
            log_r = abs(np.log(r))
            c1, c2 = 1.0, 2.0
            lower = c1 / np.sqrt(n)
            upper = c2 / np.sqrt(n)
            w = np.clip((log_r - lower) / (upper - lower), 0, 1)
            weights.append(w)
        lw = 2.5 if n == 500 else 1.5
        ax1.plot(ratios, weights, ls, linewidth=lw, label=f'n={n}')

    ax1.axvline(1.0, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('SE ratio ($R = SE_{HC3} / SE_{naive}$)')
    ax1.set_ylabel('Blending weight $w$')
    ax1.set_title('A. Weight vs SE Ratio')
    ax1.legend(title='Sample size')

    # Panel B: w vs n for fixed SE ratios
    n_values = np.logspace(np.log10(30), np.log10(10000), 100)
    for ratio, color in [(1.05, '#4DAF4A'), (1.15, '#FF7F00'), (1.5, '#E41A1C'), (2.0, '#984EA3')]:
        weights = []
        for n in n_values:
            log_r = abs(np.log(ratio))
            c1, c2 = 1.0, 2.0
            lower = c1 / np.sqrt(n)
            upper = c2 / np.sqrt(n)
            w = np.clip((log_r - lower) / (upper - lower), 0, 1)
            weights.append(w)
        ax2.plot(n_values, weights, linewidth=2, color=color, label=f'R={ratio}')

    ax2.set_xscale('log')
    ax2.set_xlabel('Sample size $n$')
    ax2.set_ylabel('Blending weight $w$')
    ax2.set_title('B. Weight vs Sample Size')
    ax2.legend(title='SE ratio')

    fig.suptitle('AMRI v2 Soft-Threshold Blending Weight', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig4_blending_weight.png')
    fig.savefig(FIGURES_DIR / 'fig4_blending_weight.pdf')
    plt.close(fig)
    print("  Fig 4: Blending Weight saved")


# ============================================================================
# FIGURE 5: Efficiency Under Correct Specification
# ============================================================================
def fig_efficiency(thm_data):
    """Show AMRI width overhead under correct specification vanishes with n."""
    df3 = thm_data.get('theorem3')
    if df3 is None:
        print("  No theorem3 data, skipping Fig 5")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel A: Width ratio vs n
    ax1.plot(df3['n'], df3['mean_width_ratio'], 'o-', color=COLORS['AMRI_v2'],
             linewidth=2, markersize=8, label='Mean width ratio')
    ax1.fill_between(df3['n'],
                     df3['mean_width_ratio'] - 0.005,
                     df3['mean_width_ratio'] + 0.005,
                     alpha=0.2, color=COLORS['AMRI_v2'])
    ax1.axhline(1.0, color='black', linestyle='--', linewidth=0.8, label='Oracle (ratio = 1)')
    ax1.set_xscale('log')
    ax1.set_xlabel('Sample size $n$')
    ax1.set_ylabel('Width ratio (AMRI / Naive)')
    ax1.set_title('A. Width Overhead Under Correct Spec')
    ax1.legend()

    # Panel B: Overhead percentage vs n
    ax2.bar(range(len(df3)), df3['overhead_pct'], color=COLORS['AMRI_v2'], alpha=0.7)
    ax2.set_xticks(range(len(df3)))
    ax2.set_xticklabels([str(n) for n in df3['n']])
    ax2.set_xlabel('Sample size $n$')
    ax2.set_ylabel('Width overhead (%)')
    ax2.set_title('B. Overhead Vanishes with $n$')

    for i, (n, oh) in enumerate(zip(df3['n'], df3['overhead_pct'])):
        ax2.text(i, oh + 0.02, f'{oh:.2f}%', ha='center', fontsize=9)

    fig.suptitle('Theorem 3: Efficiency Under Correct Specification', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig5_efficiency.png')
    fig.savefig(FIGURES_DIR / 'fig5_efficiency.pdf')
    plt.close(fig)
    print("  Fig 5: Efficiency saved")


# ============================================================================
# FIGURE S1: Per-DGP Coverage Breakdown
# ============================================================================
def fig_per_dgp_breakdown(df):
    """Supplementary: detailed coverage by DGP."""
    if df is None:
        print("  No competitor data, skipping Fig S1")
        return

    dgps = sorted(df['dgp'].unique())
    methods = ['Naive_OLS', 'Sandwich_HC3', 'AKS_Adaptive', 'AMRI_v2']
    n_dgps = len(dgps)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True)
    axes = axes.flatten()

    for idx, dgp in enumerate(dgps):
        if idx >= 6:
            break
        ax = axes[idx]
        dgp_data = df[df['dgp'] == dgp]

        for method in methods:
            mdata = dgp_data[dgp_data['method'] == method]
            if len(mdata) == 0:
                continue

            for n_val in sorted(mdata['n'].unique()):
                ndata = mdata[mdata['n'] == n_val].sort_values('delta')
                ls = '-' if n_val >= 500 else '--'
                alpha = 1.0 if n_val >= 500 else 0.5
                if method == 'AMRI_v2':
                    ax.plot(ndata['delta'], ndata['coverage'], ls,
                            color=COLORS.get(method, '#333'),
                            alpha=alpha, linewidth=1.5)

            # Average line
            avg = mdata.groupby('delta')['coverage'].mean()
            ax.plot(avg.index, avg.values, 'o-',
                    color=COLORS.get(method, '#333'),
                    linewidth=2, markersize=5,
                    label=method.replace('_', ' '))

        ax.axhline(0.95, color='black', linestyle='--', linewidth=0.8)
        ax.axhspan(0.93, 0.97, alpha=0.05, color='green')
        ax.set_title(dgp.replace('_', ' '))
        ax.set_xlabel(r'$\delta$')
        if idx % 3 == 0:
            ax.set_ylabel('Coverage')
        ax.set_ylim(0.65, 1.0)

    # Hide unused axes
    for idx in range(n_dgps, 6):
        axes[idx].set_visible(False)

    axes[0].legend(loc='lower left', fontsize=8)
    fig.suptitle('Coverage by DGP (Per-Sample-Size Detail)', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'figS1_per_dgp_breakdown.png')
    fig.savefig(FIGURES_DIR / 'figS1_per_dgp_breakdown.pdf')
    plt.close(fig)
    print("  Fig S1: Per-DGP Breakdown saved")


# ============================================================================
# FIGURE S2: Width Comparison
# ============================================================================
def fig_width_comparison(df):
    """Supplementary: CI width comparison across methods."""
    if df is None or 'avg_width' not in df.columns:
        print("  No width data, skipping Fig S2")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    dgps = sorted(df['dgp'].unique())[:3]
    methods = ['Naive_OLS', 'Sandwich_HC3', 'AKS_Adaptive', 'AMRI_v2']

    for idx, dgp in enumerate(dgps):
        ax = axes[idx]
        dgp_data = df[df['dgp'] == dgp]

        for method in methods:
            mdata = dgp_data[dgp_data['method'] == method]
            if len(mdata) == 0:
                continue
            avg = mdata.groupby('delta')['avg_width'].mean()
            ax.plot(avg.index, avg.values, 'o-',
                    color=COLORS.get(method, '#333'),
                    linewidth=2 if method == 'AMRI_v2' else 1.5,
                    label=method.replace('_', ' '))

        ax.set_xlabel(r'$\delta$')
        ax.set_title(dgp.replace('_', ' '))

    axes[0].set_ylabel('Average CI width')
    axes[0].legend(loc='upper left', fontsize=8)
    fig.suptitle('CI Width Under Increasing Misspecification', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'figS2_width_comparison.png')
    fig.savefig(FIGURES_DIR / 'figS2_width_comparison.pdf')
    plt.close(fig)
    print("  Fig S2: Width Comparison saved")


# ============================================================================
# FIGURE S3: Coverage Convergence with n
# ============================================================================
def fig_coverage_convergence(thm_data):
    """Supplementary: coverage converges to nominal as n grows."""
    df2 = thm_data.get('theorem2')
    if df2 is None:
        print("  No theorem2 data, skipping Fig S3")
        return

    dgps = df2['dgp'].unique()
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
    axes = axes.flatten()

    for idx, dgp in enumerate(dgps):
        if idx >= 6:
            break
        ax = axes[idx]
        ddata = df2[df2['dgp'] == dgp].sort_values('n')

        ax.errorbar(ddata['n'], ddata['cov_v2'],
                    yerr=[ddata['cov_v2'] - ddata['ci_lo'], ddata['ci_hi'] - ddata['cov_v2']],
                    fmt='o-', color=COLORS['AMRI_v2'], linewidth=2, markersize=6,
                    capsize=3, label='AMRI v2')

        ax.plot(ddata['n'], ddata['cov_naive'], 's--',
                color=COLORS['Naive_OLS'], linewidth=1.5, markersize=5, label='Naive')
        ax.plot(ddata['n'], ddata['cov_hc3'], '^--',
                color=COLORS['Sandwich_HC3'], linewidth=1.5, markersize=5, label='HC3')

        ax.axhline(0.95, color='black', linestyle='--', linewidth=0.8)
        ax.set_xscale('log')
        ax.set_title(dgp, fontsize=10)
        ax.set_xlabel('$n$')
        if idx % 3 == 0:
            ax.set_ylabel('Coverage')
        ax.set_ylim(0.85, 1.0)

    for idx in range(len(dgps), 6):
        axes[idx].set_visible(False)

    axes[0].legend(fontsize=8)
    fig.suptitle('Coverage Convergence (Theorem 2)', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'figS3_coverage_convergence.png')
    fig.savefig(FIGURES_DIR / 'figS3_coverage_convergence.pdf')
    plt.close(fig)
    print("  Fig S3: Coverage Convergence saved")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  GENERATING PUBLICATION FIGURES")
    print("=" * 60)
    print()

    # Load data
    comp_df = load_competitor_data()
    real_df = load_real_data()
    thm_data = load_theorem_data()

    print(f"  Competitor data: {'loaded' if comp_df is not None else 'NOT FOUND'}"
          f"{f' ({len(comp_df)} rows)' if comp_df is not None else ''}")
    print(f"  Real data: {'loaded' if real_df is not None else 'NOT FOUND'}"
          f"{f' ({len(real_df)} rows)' if real_df is not None else ''}")
    print(f"  Theorem data: {list(thm_data.keys())}")
    print()

    # Generate figures
    print("Generating main paper figures:")
    fig_coverage_vs_delta(comp_df)
    fig_amri_vs_aks(comp_df)
    fig_real_data_dashboard(real_df)
    fig_blending_weight()
    fig_efficiency(thm_data)

    print("\nGenerating supplement figures:")
    fig_per_dgp_breakdown(comp_df)
    fig_width_comparison(comp_df)
    fig_coverage_convergence(thm_data)

    # List generated files
    print(f"\nGenerated files in {FIGURES_DIR}:")
    for f in sorted(FIGURES_DIR.glob("fig*.png")):
        print(f"  {f.name}")

    print("\nDONE.")
