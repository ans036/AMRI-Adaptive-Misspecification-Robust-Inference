"""
Analysis & Visualization for Misspecification Simulation Results
================================================================
Generates publication-quality figures and summary tables.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
import json

# Style
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})

COLORS = {
    'Naive_OLS': '#d62728',       # red
    'Sandwich_HC0': '#ff7f0e',    # orange
    'Sandwich_HC3': '#2ca02c',    # green
    'Pairs_Bootstrap': '#1f77b4', # blue
    'Wild_Bootstrap': '#9467bd',  # purple
    'Bootstrap_t': '#8c564b',     # brown
    'Split_Conformal': '#e377c2', # pink
    'Jackknife': '#7f7f7f',       # gray
    'Bayesian_Normal': '#bcbd22', # olive
    'AMRI': '#17becf',            # cyan (our method)
    'Naive_GLM': '#d62728',
    'Sandwich_GLM': '#2ca02c',
}

METHOD_ORDER = [
    'Naive_OLS', 'Bayesian_Normal', 'Sandwich_HC0', 'Sandwich_HC3',
    'Pairs_Bootstrap', 'Wild_Bootstrap', 'Bootstrap_t',
    'Split_Conformal', 'Jackknife', 'AMRI'
]

DGP_LABELS = {
    'DGP1_nonlinearity': 'DGP1: Nonlinearity',
    'DGP2_heavy_tails': 'DGP2: Heavy Tails',
    'DGP3_heteroscedasticity': 'DGP3: Heteroscedasticity',
    'DGP4_omitted_variable': 'DGP4: Omitted Variable',
    'DGP5_link_function': 'DGP5: Wrong Link',
    'DGP6_clustering': 'DGP6: Clustering',
}

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def load_results():
    """Load all simulation result files."""
    results_dir = Path(__file__).resolve().parent.parent / "results"
    dfs = []
    for f in results_dir.glob("results_*.csv"):
        df = pd.read_csv(f)
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError("No results files found!")
    combined = pd.concat(dfs, ignore_index=True)
    # Remove exact duplicates
    combined = combined.drop_duplicates(subset=['dgp', 'delta', 'n', 'method'])
    return combined


# ============================================================================
# FIGURE 1: Coverage vs Severity (Main Result)
# ============================================================================

def plot_coverage_vs_severity(df, figsize=(18, 10)):
    """Plot coverage probability vs misspecification severity for each DGP."""
    dgps = sorted(df['dgp'].unique())
    n_dgps = len(dgps)
    ncols = min(3, n_dgps)
    nrows = (n_dgps + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for idx, dgp in enumerate(dgps):
        row, col = idx // ncols, idx % ncols
        ax = axes[row][col]
        sub = df[(df['dgp'] == dgp) & (df['n'] == 500)]  # Fix n=500

        methods_present = [m for m in METHOD_ORDER if m in sub['method'].unique()]

        for method in methods_present:
            msub = sub[sub['method'] == method].sort_values('delta')
            ax.plot(msub['delta'], msub['coverage'],
                    'o-', color=COLORS.get(method, 'black'),
                    label=method.replace('_', ' '), linewidth=2, markersize=5)
            # Add MC SE band
            ax.fill_between(msub['delta'],
                            msub['coverage'] - 1.96 * msub['coverage_mc_se'],
                            msub['coverage'] + 1.96 * msub['coverage_mc_se'],
                            alpha=0.1, color=COLORS.get(method, 'black'))

        ax.axhline(0.95, color='black', linestyle='--', linewidth=1, alpha=0.7)
        ax.axhspan(0.94, 0.96, alpha=0.1, color='green')
        ax.set_xlabel('Misspecification Severity (δ)')
        ax.set_ylabel('Coverage Probability')
        ax.set_title(DGP_LABELS.get(dgp, dgp))
        ax.set_ylim(0.5, 1.01)
        ax.set_xlim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    # Remove unused subplots
    for idx in range(n_dgps, nrows * ncols):
        row, col = idx // ncols, idx % ncols
        axes[row][col].set_visible(False)

    # Single legend
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, frameon=True, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('Coverage Probability vs. Misspecification Severity (n=500, nominal=95%)',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig1_coverage_vs_severity.png', bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'fig1_coverage_vs_severity.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig1_coverage_vs_severity.png")


# ============================================================================
# FIGURE 2: Coverage vs Sample Size
# ============================================================================

def plot_coverage_vs_n(df, figsize=(18, 10)):
    """Plot coverage vs sample size, for severe misspecification (delta=1.0)."""
    dgps = sorted(df['dgp'].unique())
    n_dgps = len(dgps)
    ncols = min(3, n_dgps)
    nrows = (n_dgps + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for idx, dgp in enumerate(dgps):
        row, col = idx // ncols, idx % ncols
        ax = axes[row][col]
        sub = df[(df['dgp'] == dgp) & (df['delta'] == 1.0)]

        methods_present = [m for m in METHOD_ORDER if m in sub['method'].unique()]

        for method in methods_present:
            msub = sub[sub['method'] == method].sort_values('n')
            ax.plot(msub['n'], msub['coverage'],
                    'o-', color=COLORS.get(method, 'black'),
                    label=method.replace('_', ' '), linewidth=2, markersize=5)

        ax.axhline(0.95, color='black', linestyle='--', linewidth=1, alpha=0.7)
        ax.axhspan(0.94, 0.96, alpha=0.1, color='green')
        ax.set_xlabel('Sample Size (n)')
        ax.set_ylabel('Coverage Probability')
        ax.set_title(DGP_LABELS.get(dgp, dgp))
        ax.set_ylim(0.4, 1.01)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)

    for idx in range(n_dgps, nrows * ncols):
        row, col = idx // ncols, idx % ncols
        axes[row][col].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, frameon=True, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('Coverage vs. Sample Size Under Severe Misspecification (δ=1.0, nominal=95%)',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig2_coverage_vs_n.png', bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'fig2_coverage_vs_n.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig2_coverage_vs_n.png")


# ============================================================================
# FIGURE 3: Efficiency-Robustness Frontier
# ============================================================================

def plot_efficiency_robustness(df, figsize=(14, 8)):
    """Scatter: coverage vs interval width for each method (at delta=0.5, n=500)."""
    sub = df[(df['delta'] == 0.5) & (df['n'] == 500)]
    dgps = sorted(sub['dgp'].unique())

    fig, axes = plt.subplots(1, len(dgps), figsize=figsize, squeeze=False)

    for idx, dgp in enumerate(dgps):
        ax = axes[0][idx]
        dsub = sub[sub['dgp'] == dgp]

        for _, row in dsub.iterrows():
            method = row['method']
            ax.scatter(row['avg_width'], row['coverage'],
                       c=COLORS.get(method, 'black'), s=120, zorder=5,
                       edgecolors='black', linewidth=0.5)
            ax.annotate(method.replace('_', '\n'), (row['avg_width'], row['coverage']),
                        fontsize=7, ha='center', va='bottom',
                        xytext=(0, 8), textcoords='offset points')

        ax.axhline(0.95, color='black', linestyle='--', alpha=0.5)
        ax.axhspan(0.94, 0.96, alpha=0.08, color='green')
        ax.set_xlabel('Average Interval Width')
        ax.set_ylabel('Coverage Probability')
        ax.set_title(DGP_LABELS.get(dgp, dgp), fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Efficiency-Robustness Frontier (δ=0.5, n=500)\nIdeal: top-left (high coverage, narrow width)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig3_efficiency_robustness.png', bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'fig3_efficiency_robustness.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig3_efficiency_robustness.png")


# ============================================================================
# FIGURE 4: Heatmap of Coverage
# ============================================================================

def plot_coverage_heatmap(df, figsize=(20, 10)):
    """Heatmap: methods × (DGP, delta) showing coverage."""
    sub = df[df['n'] == 500].copy()
    sub['scenario'] = sub['dgp'].map(lambda x: x.split('_', 1)[0]) + '\nδ=' + sub['delta'].astype(str)
    pivot = sub.pivot_table(values='coverage', index='method', columns='scenario')

    # Reorder methods
    method_order = [m for m in METHOD_ORDER if m in pivot.index]
    pivot = pivot.reindex(method_order)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0.95,
                vmin=0.5, vmax=1.0, ax=ax, linewidths=0.5,
                cbar_kws={'label': 'Coverage Probability'})
    ax.set_title('Coverage Heatmap: Methods × Scenarios (n=500, nominal=0.95)',
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Method')
    ax.set_xlabel('Scenario')
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig4_coverage_heatmap.png', bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'fig4_coverage_heatmap.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig4_coverage_heatmap.png")


# ============================================================================
# FIGURE 5: Coverage Degradation Gradient
# ============================================================================

def plot_coverage_degradation(df, figsize=(14, 6)):
    """Show how fast coverage degrades for each method as delta increases."""
    sub = df[df['n'] == 500]
    methods_present = [m for m in METHOD_ORDER if m in sub['method'].unique()]

    fig, ax = plt.subplots(figsize=figsize)

    for method in methods_present:
        msub = sub[sub['method'] == method]
        # Average coverage across all DGPs for each delta
        avg = msub.groupby('delta')['coverage'].mean().reset_index()
        avg = avg.sort_values('delta')
        ax.plot(avg['delta'], avg['coverage'], 'o-',
                color=COLORS.get(method, 'black'),
                label=method.replace('_', ' '), linewidth=2.5, markersize=7)

    ax.axhline(0.95, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Nominal (95%)')
    ax.axhspan(0.93, 0.97, alpha=0.08, color='green')
    ax.set_xlabel('Misspecification Severity (δ)', fontsize=13)
    ax.set_ylabel('Average Coverage (across all DGPs)', fontsize=13)
    ax.set_title('Coverage Degradation: How Fast Does Each Method Break?',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0.55, 1.01)
    ax.legend(loc='lower left', frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig5_coverage_degradation.png', bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'fig5_coverage_degradation.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig5_coverage_degradation.png")


# ============================================================================
# FIGURE 6: RMSE Comparison
# ============================================================================

def plot_rmse_comparison(df, figsize=(16, 8)):
    """Bar chart of RMSE by method for each DGP at delta=0.5, n=500."""
    sub = df[(df['delta'] == 0.5) & (df['n'] == 500)]
    dgps = sorted(sub['dgp'].unique())

    fig, axes = plt.subplots(1, len(dgps), figsize=figsize, sharey=True, squeeze=False)

    for idx, dgp in enumerate(dgps):
        ax = axes[0][idx]
        dsub = sub[sub['dgp'] == dgp].sort_values('rmse')
        colors = [COLORS.get(m, 'black') for m in dsub['method']]
        bars = ax.barh(dsub['method'].str.replace('_', ' '), dsub['rmse'], color=colors)
        ax.set_xlabel('RMSE')
        ax.set_title(DGP_LABELS.get(dgp, dgp).split(': ')[1], fontsize=10)
        ax.grid(True, alpha=0.3, axis='x')

    fig.suptitle('Root Mean Square Error by Method (δ=0.5, n=500)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig6_rmse_comparison.png', bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'fig6_rmse_comparison.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig6_rmse_comparison.png")


# ============================================================================
# FIGURE 7: The "SE Ratio" Diagnostic
# ============================================================================

def plot_se_ratio_analysis(df, figsize=(14, 6)):
    """Plot avg_width ratio between sandwich and naive as misspecification diagnostic."""
    sub = df[df['n'] == 500]
    naive = sub[sub['method'] == 'Naive_OLS'][['dgp', 'delta', 'avg_width']].rename(
        columns={'avg_width': 'width_naive'})
    sandwich = sub[sub['method'] == 'Sandwich_HC3'][['dgp', 'delta', 'avg_width']].rename(
        columns={'avg_width': 'width_sandwich'})

    merged = naive.merge(sandwich, on=['dgp', 'delta'])
    merged['se_ratio'] = merged['width_sandwich'] / merged['width_naive']

    fig, ax = plt.subplots(figsize=figsize)
    dgps = sorted(merged['dgp'].unique())
    for dgp in dgps:
        dsub = merged[merged['dgp'] == dgp].sort_values('delta')
        ax.plot(dsub['delta'], dsub['se_ratio'], 'o-',
                label=DGP_LABELS.get(dgp, dgp).split(': ')[1], linewidth=2, markersize=6)

    ax.axhline(1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Misspecification Severity (δ)', fontsize=13)
    ax.set_ylabel('SE Ratio (Sandwich / Naive)', fontsize=13)
    ax.set_title('Misspecification Diagnostic: Sandwich-to-Naive SE Ratio\n(Ratio ≈ 1 → no misspecification; Ratio >> 1 → severe misspecification)',
                 fontsize=13, fontweight='bold')
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig7_se_ratio_diagnostic.png', bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'fig7_se_ratio_diagnostic.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig7_se_ratio_diagnostic.png")


# ============================================================================
# SUMMARY TABLE
# ============================================================================

def generate_summary_table(df):
    """Generate a publication-quality summary table."""
    sub = df[df['n'] == 500]
    table = sub.pivot_table(
        values=['coverage', 'avg_width', 'bias', 'rmse'],
        index='method',
        columns=['dgp', 'delta'],
        aggfunc='mean'
    )

    # Save full table
    table.to_csv(FIGURES_DIR / 'summary_table_full.csv')

    # Condensed table: just coverage at delta = 0.0, 0.5, 1.0
    deltas_show = [0.0, 0.5, 1.0]
    sub2 = sub[sub['delta'].isin(deltas_show)]
    coverage_table = sub2.pivot_table(values='coverage', index='method',
                                       columns=['dgp', 'delta'], aggfunc='mean')
    coverage_table.to_csv(FIGURES_DIR / 'coverage_table.csv')
    print(f"  Saved: summary_table_full.csv, coverage_table.csv")
    return coverage_table


# ============================================================================
# STATISTICAL ANALYSIS: Hypothesis Testing
# ============================================================================

def statistical_analysis(df):
    """Compute formal statistical tests on the results."""
    results = {}

    # Test 1: For each method, is coverage significantly below 0.95 under misspecification?
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS OF COVERAGE")
    print("=" * 80)

    sub = df[(df['n'] == 500) & (df['delta'] == 1.0)]

    print(f"\n{'Method':<20} {'Mean Cov':>10} {'MC SE':>10} {'z-stat':>10} {'p-value':>10} {'Sig?':>6}")
    print("-" * 70)

    for method in sorted(sub['method'].unique()):
        msub = sub[sub['method'] == method]
        mean_cov = msub['coverage'].mean()
        # MC SE of the mean coverage across DGPs
        se_cov = msub['coverage'].std() / np.sqrt(len(msub))
        # Test: H0: true coverage >= 0.95
        z = (mean_cov - 0.95) / (se_cov + 1e-10)
        from scipy.stats import norm
        p = norm.cdf(z)  # one-sided: is coverage significantly below 0.95?
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"{method:<20} {mean_cov:>10.4f} {se_cov:>10.4f} {z:>10.3f} {p:>10.4f} {sig:>6}")
        results[method] = {'mean_coverage': mean_cov, 'se': se_cov, 'z': z, 'p': p}

    # Test 2: Pairwise comparison of AMRI vs each other method
    if 'AMRI' in sub['method'].values:
        print(f"\n\nPAIRWISE COMPARISONS: AMRI vs. each method (at d=1.0, n=500)")
        print("-" * 70)
        amri_coverages = sub[sub['method'] == 'AMRI']['coverage'].values

        for method in sorted(sub['method'].unique()):
            if method == 'AMRI':
                continue
            other_coverages = sub[sub['method'] == method]['coverage'].values
            # Match by DGP
            amri_df = sub[sub['method'] == 'AMRI'][['dgp', 'coverage']].rename(columns={'coverage': 'cov_amri'})
            other_df = sub[sub['method'] == method][['dgp', 'coverage']].rename(columns={'coverage': 'cov_other'})
            merged = amri_df.merge(other_df, on='dgp')
            diff = merged['cov_amri'] - merged['cov_other']
            mean_diff = diff.mean()
            se_diff = diff.std() / np.sqrt(len(diff))
            t = mean_diff / (se_diff + 1e-10)
            from scipy.stats import t as t_dist
            p = 1 - t_dist.cdf(abs(t), len(diff) - 1)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  AMRI vs {method:<20} Δcoverage = {mean_diff:+.4f}  t = {t:6.3f}  p = {p:.4f} {sig}")

    # Save
    with open(FIGURES_DIR / 'statistical_tests.json', 'w') as f:
        json.dump({k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()}, f, indent=2)
    print(f"\n  Saved: statistical_tests.json")

    return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("Loading results...")
    df = load_results()
    print(f"Loaded {len(df)} scenarios from {df['dgp'].nunique()} DGPs, "
          f"{df['method'].nunique()} methods, {df['n'].nunique()} sample sizes")

    print(f"\nDGPs: {sorted(df['dgp'].unique())}")
    print(f"Methods: {sorted(df['method'].unique())}")
    print(f"Deltas: {sorted(df['delta'].unique())}")
    print(f"Sample sizes: {sorted(df['n'].unique())}")

    print("\nGenerating figures...")
    plot_coverage_vs_severity(df)
    plot_coverage_vs_n(df)

    # Only plot these if we have enough methods
    if df['method'].nunique() >= 3:
        plot_efficiency_robustness(df)
        plot_coverage_heatmap(df)
        plot_coverage_degradation(df)
        plot_rmse_comparison(df)

    if 'Naive_OLS' in df['method'].values and 'Sandwich_HC3' in df['method'].values:
        plot_se_ratio_analysis(df)

    print("\nGenerating summary table...")
    cov_table = generate_summary_table(df)
    print("\nCoverage Table (n=500):")
    print(cov_table.to_string())

    print("\nRunning statistical analysis...")
    stats_results = statistical_analysis(df)

    print("\n" + "=" * 80)
    print("ALL ANALYSIS COMPLETE")
    print(f"Figures saved to: {FIGURES_DIR}")
    print("=" * 80)
