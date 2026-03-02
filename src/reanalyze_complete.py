"""
Complete Reanalysis Script
==========================
Run this after the full simulation completes to regenerate all figures,
re-run all statistical tests, and validate hypotheses across ALL 6 DGPs.

Usage: python -u src/reanalyze_complete.py
"""
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.family': 'serif',
})

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGS = Path(__file__).resolve().parent.parent / "figures"
FIGS.mkdir(exist_ok=True)

COLORS = {
    'Naive_OLS': '#d62728', 'Sandwich_HC0': '#ff7f0e', 'Sandwich_HC3': '#2ca02c',
    'Pairs_Bootstrap': '#1f77b4', 'Wild_Bootstrap': '#9467bd', 'Bootstrap_t': '#8c564b',
    'Bayesian': '#bcbd22', 'AMRI': '#17becf',
}

METHOD_LABELS = {
    'Naive_OLS': 'Naive OLS', 'Sandwich_HC0': 'Sandwich HC0', 'Sandwich_HC3': 'Sandwich HC3',
    'Pairs_Bootstrap': 'Pairs Bootstrap', 'Wild_Bootstrap': 'Wild Bootstrap',
    'Bootstrap_t': 'Bootstrap-t', 'Bayesian': 'Bayesian', 'AMRI': 'AMRI (Proposed)',
}

DGP_LABELS = {
    'DGP1_nonlinearity': 'Nonlinearity',
    'DGP2_heavy_tails': 'Heavy Tails',
    'DGP3_heteroscedastic': 'Heteroscedasticity',
    'DGP4_omitted_var': 'Omitted Variable',
    'DGP5_clustering': 'Clustering',
    'DGP6_contaminated': 'Contamination',
}


def load_all():
    dfs = []
    for f in RESULTS_DIR.glob("*.csv"):
        dfs.append(pd.read_csv(f))
    df = pd.concat(dfs, ignore_index=True)
    if 'B_valid' in df.columns and 'B' not in df.columns:
        df['B'] = df['B_valid']
    elif 'B_valid' in df.columns:
        df['B'] = df['B'].fillna(df['B_valid'])
    df['method'] = df['method'].replace({'Bayesian_Normal': 'Bayesian'})
    df = df.drop_duplicates(subset=['dgp', 'delta', 'n', 'method'], keep='last')
    df = df.dropna(subset=['coverage'])
    return df


# ============================================================================
# FIGURE 1: Coverage Degradation per DGP (3x2 grid)
# ============================================================================
def fig1_coverage_per_dgp(df, n_val=500):
    dgps = sorted(df['dgp'].unique())
    n_dgps = len(dgps)
    ncols = min(3, n_dgps)
    nrows = (n_dgps + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7*ncols, 6*nrows), squeeze=False)

    methods_order = ['Naive_OLS', 'Bayesian', 'Sandwich_HC0', 'Sandwich_HC3',
                     'Pairs_Bootstrap', 'Wild_Bootstrap', 'Bootstrap_t', 'AMRI']

    for idx, dgp in enumerate(dgps):
        ax = axes[idx // ncols][idx % ncols]
        sub = df[(df['dgp'] == dgp) & (df['n'] == n_val)]

        for method in methods_order:
            msub = sub[sub['method'] == method]
            if len(msub) == 0:
                continue
            avg = msub.sort_values('delta')
            style = '--' if method in ['Naive_OLS', 'Bayesian'] else '-'
            lw = 3 if method == 'AMRI' else 1.5
            alpha = 0.9 if method in ['AMRI', 'Naive_OLS', 'Sandwich_HC3'] else 0.5
            ax.plot(avg['delta'], avg['coverage'], f'o{style}',
                    color=COLORS.get(method, 'black'),
                    label=METHOD_LABELS.get(method, method),
                    linewidth=lw, markersize=5, alpha=alpha)

        ax.axhline(0.95, color='black', linestyle=':', linewidth=1, alpha=0.5)
        ax.axhspan(0.935, 0.965, alpha=0.04, color='green')
        ax.set_xlabel('Misspecification Severity (delta)')
        ax.set_ylabel('Coverage Probability')
        ax.set_title(DGP_LABELS.get(dgp, dgp), fontsize=13, fontweight='bold')
        ax.set_ylim(0.5, 1.01)
        ax.grid(True, alpha=0.15)

    # Single legend
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    # Remove empty subplots
    for idx in range(n_dgps, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(f'Coverage Degradation by DGP Type (n={n_val}, 2000 reps)\n'
                 f'Dashed = model-based; Solid = robust; Thick cyan = AMRI',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGS / 'FINAL_1_coverage_per_dgp.png', bbox_inches='tight')
    plt.close()
    print("  Saved: FINAL_1_coverage_per_dgp.png")


# ============================================================================
# FIGURE 2: AMRI Mechanism + Efficiency-Robustness Frontier
# ============================================================================
def fig2_amri_mechanism(df, n_val=500):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel A: SE ratio diagnostic
    ax = axes[0]
    sub = df[df['n'] == n_val]
    naive_w = sub[sub['method'] == 'Naive_OLS'][['dgp', 'delta', 'avg_width']].rename(
        columns={'avg_width': 'w_naive'})
    hc3_w = sub[sub['method'] == 'Sandwich_HC3'][['dgp', 'delta', 'avg_width']].rename(
        columns={'avg_width': 'w_hc3'})
    merged = naive_w.merge(hc3_w, on=['dgp', 'delta'])
    merged['se_ratio'] = merged['w_hc3'] / merged['w_naive']

    for dgp in sorted(merged['dgp'].unique()):
        dsub = merged[merged['dgp'] == dgp].sort_values('delta')
        label = DGP_LABELS.get(dgp, dgp)
        ax.plot(dsub['delta'], dsub['se_ratio'], 'o-', linewidth=2, markersize=6, label=label)

    ax.axhline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    thresh = 1 + 2 / np.sqrt(n_val)
    ax.axhline(thresh, color='red', linestyle=':', alpha=0.3)
    ax.text(0.02, thresh + 0.02, f'AMRI threshold\n(n={n_val})', fontsize=8, color='red')
    ax.set_xlabel('Misspecification Severity (delta)')
    ax.set_ylabel('SE Ratio (HC3 / Naive)')
    ax.set_title('A: Misspecification Diagnostic\n(SE Ratio as Signal)', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Panel B: Coverage comparison (Naive vs HC3 vs AMRI)
    ax = axes[1]
    for method in ['Naive_OLS', 'Sandwich_HC3', 'AMRI']:
        msub = sub[sub['method'] == method]
        avg = msub.groupby('delta')['coverage'].mean().reset_index().sort_values('delta')
        style = '--' if method == 'Naive_OLS' else '-'
        lw = 3 if method == 'AMRI' else 2
        ax.plot(avg['delta'], avg['coverage'], f'o{style}',
                color=COLORS[method], label=METHOD_LABELS[method],
                linewidth=lw, markersize=7)

    ax.axhline(0.95, color='black', linestyle=':', linewidth=1)
    ax.set_xlabel('Misspecification Severity (delta)')
    ax.set_ylabel('Coverage Probability')
    ax.set_title('B: AMRI Adapts to\nMisspecification Level', fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0.5, 1.01)
    ax.grid(True, alpha=0.2)

    # Panel C: Efficiency frontier at delta=0 and delta=1
    ax = axes[2]
    for delta_val, marker, label_suffix in [(0.0, 'o', 'd=0'), (1.0, 's', 'd=1')]:
        sub_d = df[(df['delta'] == delta_val) & (df['n'] == n_val)]
        agg = sub_d.groupby('method').agg({'coverage': 'mean', 'avg_width': 'mean'}).reset_index()
        for _, row in agg.iterrows():
            method = row['method']
            size = 200 if method == 'AMRI' else 80
            ax.scatter(row['avg_width'], row['coverage'],
                       c=COLORS.get(method, 'gray'), s=size, marker=marker,
                       edgecolors='black', linewidth=0.5, alpha=0.8,
                       label=f'{METHOD_LABELS.get(method, method)} ({label_suffix})')

    ax.axhline(0.95, color='red', linestyle='--', alpha=0.3)
    ax.set_xlabel('Average Width (narrower = better)')
    ax.set_ylabel('Coverage')
    ax.set_title('C: Efficiency-Robustness\nFrontier', fontweight='bold')
    ax.grid(True, alpha=0.2)
    # Slim legend
    handles, labels = ax.get_legend_handles_labels()
    # Show only unique methods
    seen = set()
    unique_h, unique_l = [], []
    for h, l in zip(handles, labels):
        method_name = l.split(' (')[0]
        if method_name not in seen:
            seen.add(method_name)
            unique_h.append(h)
            unique_l.append(method_name)
    ax.legend(unique_h, unique_l, fontsize=7, loc='lower left')

    plt.tight_layout()
    fig.savefig(FIGS / 'FINAL_2_amri_mechanism.png', bbox_inches='tight')
    plt.close()
    print("  Saved: FINAL_2_amri_mechanism.png")


# ============================================================================
# FIGURE 3: Sample Size Paradox
# ============================================================================
def fig3_sample_size(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax_idx, delta_val in enumerate([0.5, 1.0]):
        ax = axes[ax_idx]
        sub = df[df['delta'] == delta_val]

        for method in ['Naive_OLS', 'Sandwich_HC3', 'Bootstrap_t', 'AMRI']:
            if method not in sub['method'].unique():
                continue
            msub = sub[sub['method'] == method]
            avg = msub.groupby('n')['coverage'].mean().reset_index().sort_values('n')
            style = '--' if method == 'Naive_OLS' else '-'
            lw = 3 if method == 'AMRI' else 2
            ax.plot(avg['n'], avg['coverage'], f'o{style}',
                    color=COLORS[method], label=METHOD_LABELS[method],
                    linewidth=lw, markersize=6)

        ax.axhline(0.95, color='black', linestyle=':', linewidth=1)
        ax.set_xlabel('Sample Size (n)')
        ax.set_ylabel('Coverage')
        ax.set_title(f'delta = {delta_val}', fontsize=13, fontweight='bold')
        ax.set_xscale('log')
        ax.set_ylim(0.5, 1.01)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.2)

    fig.suptitle('The Sample Size Paradox: More Data WORSENS Naive Inference\n'
                 'Under Misspecification (robust methods unaffected)',
                 fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    fig.savefig(FIGS / 'FINAL_3_sample_size_paradox.png', bbox_inches='tight')
    plt.close()
    print("  Saved: FINAL_3_sample_size_paradox.png")


# ============================================================================
# FIGURE 4: Grand Heatmap
# ============================================================================
def fig4_heatmap(df):
    methods_order = ['Naive_OLS', 'Bayesian', 'Sandwich_HC0', 'Sandwich_HC3',
                     'Pairs_Bootstrap', 'Wild_Bootstrap', 'Bootstrap_t', 'AMRI']
    methods_present = [m for m in methods_order if m in df['method'].unique()]

    # Average across DGPs
    agg = df.groupby(['method', 'delta', 'n'])['coverage'].mean().reset_index()
    agg['scenario'] = 'd=' + agg['delta'].astype(str) + ', n=' + agg['n'].astype(str)

    pivot = agg.pivot_table(values='coverage', index='method', columns='scenario')
    cols_order = sorted(pivot.columns,
                        key=lambda x: (float(x.split(',')[0].split('=')[1]),
                                       float(x.split(',')[1].split('=')[1])))
    pivot = pivot.reindex(methods_present)[cols_order]

    fig, ax = plt.subplots(figsize=(22, 7))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0.95,
                vmin=0.5, vmax=1.0, ax=ax, linewidths=0.5,
                cbar_kws={'label': 'Coverage Probability'})
    ax.set_title('Coverage Heatmap: Methods x Scenarios (nominal = 0.95)\n'
                 'Green = valid | Red = failure | Averaged across all DGPs',
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Method')
    ax.set_xlabel('Scenario (delta, n)')
    plt.tight_layout()
    fig.savefig(FIGS / 'FINAL_4_heatmap.png', bbox_inches='tight')
    plt.close()
    print("  Saved: FINAL_4_heatmap.png")


# ============================================================================
# FIGURE 5: Width Tax
# ============================================================================
def fig5_width_tax(df, n_val=500):
    sub = df[df['n'] == n_val]
    naive_widths = sub[sub['method'] == 'Naive_OLS'][['dgp', 'delta', 'avg_width']].rename(
        columns={'avg_width': 'w_naive'})

    results = []
    for method in sub['method'].unique():
        if method == 'Naive_OLS':
            continue
        mdata = sub[sub['method'] == method][['dgp', 'delta', 'avg_width']].rename(
            columns={'avg_width': 'w_method'})
        merged = mdata.merge(naive_widths, on=['dgp', 'delta'])
        merged['width_ratio'] = merged['w_method'] / merged['w_naive']
        merged['method'] = method
        results.append(merged)

    if not results:
        return
    all_ratios = pd.concat(results)

    fig, ax = plt.subplots(figsize=(12, 6))
    for method in ['Sandwich_HC3', 'Pairs_Bootstrap', 'Bootstrap_t', 'AMRI']:
        msub = all_ratios[all_ratios['method'] == method]
        if len(msub) == 0:
            continue
        avg = msub.groupby('delta')['width_ratio'].agg(['mean', 'std']).reset_index()
        avg = avg.sort_values('delta')
        lw = 3 if method == 'AMRI' else 2
        ax.plot(avg['delta'], avg['mean'], 'o-',
                color=COLORS[method], label=METHOD_LABELS[method],
                linewidth=lw, markersize=7)
        if len(avg) > 1:
            ax.fill_between(avg['delta'],
                            avg['mean'] - avg['std'],
                            avg['mean'] + avg['std'],
                            color=COLORS[method], alpha=0.1)

    ax.axhline(1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.set_xlabel('Misspecification Severity (delta)', fontsize=13)
    ax.set_ylabel('Width Ratio (Method / Naive OLS)', fontsize=13)
    ax.set_title(f'The Robustness Tax at n={n_val}\n'
                 'Shaded: +/- 1 SD across DGPs', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig(FIGS / 'FINAL_5_width_tax.png', bbox_inches='tight')
    plt.close()
    print("  Saved: FINAL_5_width_tax.png")


# ============================================================================
# FIGURE 6: Per-DGP Coverage Degradation Bar Chart
# ============================================================================
def fig6_dgp_comparison(df, n_val=500):
    """Show coverage at delta=1.0 by method and DGP."""
    sub = df[(df['delta'] == 1.0) & (df['n'] == n_val)]
    if len(sub) == 0:
        print("  Skipping fig6: no data at delta=1.0, n=500")
        return

    methods_order = ['Naive_OLS', 'Bayesian', 'Sandwich_HC0', 'Sandwich_HC3',
                     'Pairs_Bootstrap', 'Wild_Bootstrap', 'Bootstrap_t', 'AMRI']
    methods_present = [m for m in methods_order if m in sub['method'].unique()]
    dgps = sorted(sub['dgp'].unique())

    x = np.arange(len(dgps))
    width = 0.8 / len(methods_present)

    fig, ax = plt.subplots(figsize=(14, 7))
    for i, method in enumerate(methods_present):
        vals = []
        for dgp in dgps:
            dsub = sub[(sub['method'] == method) & (sub['dgp'] == dgp)]
            vals.append(dsub['coverage'].values[0] if len(dsub) > 0 else np.nan)
        offset = (i - len(methods_present)/2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9,
                      color=COLORS.get(method, 'gray'),
                      label=METHOD_LABELS.get(method, method),
                      edgecolor='black', linewidth=0.3)

    ax.axhline(0.95, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Nominal 95%')
    ax.set_xlabel('Data Generating Process')
    ax.set_ylabel('Coverage Probability')
    ax.set_title(f'Coverage at Severe Misspecification (delta=1.0, n={n_val})\nBy DGP Type',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([DGP_LABELS.get(d, d) for d in dgps], rotation=30, ha='right')
    ax.legend(fontsize=9, ncol=3, loc='lower left')
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, axis='y', alpha=0.2)
    plt.tight_layout()
    fig.savefig(FIGS / 'FINAL_6_dgp_comparison.png', bbox_inches='tight')
    plt.close()
    print("  Saved: FINAL_6_dgp_comparison.png")


# ============================================================================
# STATISTICAL TESTS (from statistical_guarantees.py, condensed)
# ============================================================================
def run_all_tests(df):
    print("\n" + "=" * 80)
    print("COMPREHENSIVE STATISTICAL TESTS")
    print("=" * 80)

    methods = sorted(df['method'].unique())
    dgps = sorted(df['dgp'].unique())
    n_dgps = len(dgps)

    # Test 1: Coverage validity
    print(f"\n--- TEST 1: Coverage Validity (Fisher combined test) ---")
    print(f"  {'Method':<20} {'Avg Cov':>8} {'Min Cov':>8} {'Verdict':>10}")
    for method in methods:
        msub = df[df['method'] == method]
        p_values = []
        for _, row in msub.iterrows():
            B = int(row.get('B', row.get('B_valid', 2000)))
            cov = row['coverage']
            k = int(round(cov * B))
            p_values.append(stats.binom.cdf(k, B, 0.95))
        log_p = np.log(np.maximum(np.array(p_values), 1e-300))
        fisher_stat = -2 * log_p.sum()
        fisher_pval = 1 - stats.chi2.cdf(fisher_stat, 2 * len(p_values))
        verdict = 'VALID' if fisher_pval > 0.05 else 'FAILS'
        print(f"  {method:<20} {msub['coverage'].mean():>8.4f} {msub['coverage'].min():>8.4f} {verdict:>10}")

    # Test 2: Monotonic degradation
    print(f"\n--- TEST 2: Monotonic Degradation (Spearman) ---")
    for method in ['Naive_OLS', 'Bayesian']:
        if method not in df['method'].values:
            continue
        msub = df[df['method'] == method]
        rhos = []
        for dgp in dgps:
            for n_val in msub['n'].unique():
                dsub = msub[(msub['dgp'] == dgp) & (msub['n'] == n_val)].sort_values('delta')
                if len(dsub) >= 3:
                    rho, _ = stats.spearmanr(dsub['delta'], dsub['coverage'])
                    rhos.append(rho)
        if rhos:
            print(f"  {method}: avg_rho={np.mean(rhos):.4f}, all_negative={all(r < 0 for r in rhos)}, "
                  f"n_combos={len(rhos)}")

    # Test 3: AMRI best-of-both
    print(f"\n--- TEST 3: AMRI Best-of-Both-Worlds ---")
    d0 = df[df['delta'] == 0.0]
    d_pos = df[df['delta'] > 0]

    amri_d0 = d0[d0['method'] == 'AMRI']
    naive_d0 = d0[d0['method'] == 'Naive_OLS']
    if len(amri_d0) > 0 and len(naive_d0) > 0:
        m = amri_d0[['dgp', 'n', 'coverage', 'avg_width']].merge(
            naive_d0[['dgp', 'n', 'coverage', 'avg_width']],
            on=['dgp', 'n'], suffixes=('_amri', '_naive'))
        cov_diff = m['coverage_amri'] - m['coverage_naive']
        width_ratio = m['avg_width_amri'] / m['avg_width_naive']
        print(f"  At d=0: cov_diff={cov_diff.mean():.5f}, max_width_ratio={width_ratio.max():.4f}")

    amri_pos = d_pos[d_pos['method'] == 'AMRI']
    hc3_pos = d_pos[d_pos['method'] == 'Sandwich_HC3']
    if len(amri_pos) > 0 and len(hc3_pos) > 0:
        m = amri_pos[['dgp', 'delta', 'n', 'coverage']].merge(
            hc3_pos[['dgp', 'delta', 'n', 'coverage']],
            on=['dgp', 'delta', 'n'], suffixes=('_amri', '_hc3'))
        diff = m['coverage_amri'] - m['coverage_hc3']
        t_stat, p_val = stats.ttest_1samp(diff, 0)
        p_one = p_val / 2 if t_stat > 0 else 1 - p_val / 2
        print(f"  At d>0: AMRI-HC3={diff.mean():.5f}, t={t_stat:.3f}, p_one={p_one:.6f}, "
              f"positive={int((diff>0).sum())}/{len(diff)}")

    # Test 4: Degradation ordering
    print(f"\n--- TEST 4: Degradation Rate Ordering ---")
    method_slopes = {}
    for method in methods:
        msub = df[df['method'] == method]
        slopes = []
        for dgp in dgps:
            for n_val in msub['n'].unique():
                dsub = msub[(msub['dgp'] == dgp) & (msub['n'] == n_val)].sort_values('delta')
                if len(dsub) >= 3:
                    slope, _, _, _, _ = stats.linregress(dsub['delta'], dsub['coverage'])
                    slopes.append(slope)
        if slopes:
            method_slopes[method] = np.mean(slopes)

    for method, slope in sorted(method_slopes.items(), key=lambda x: x[1]):
        print(f"  {method:<20}: {slope:+.5f}")

    # Test 5: Generalization across DGPs
    print(f"\n--- TEST 5: Generalization Across DGPs ---")
    print(f"  DGPs tested: {n_dgps}")
    for method in ['AMRI', 'Sandwich_HC3', 'Naive_OLS']:
        if method not in df['method'].values:
            continue
        msub = df[df['method'] == method]
        per_dgp = msub.groupby('dgp')['coverage'].mean()
        print(f"  {method}:")
        for dgp, cov in per_dgp.items():
            print(f"    {dgp}: {cov:.4f}")

    # Test 6: Sample size paradox
    print(f"\n--- TEST 6: Sample Size Paradox ---")
    high_delta = df[df['delta'] >= 0.5]
    for method in ['Naive_OLS', 'Sandwich_HC3', 'AMRI']:
        if method not in high_delta['method'].values:
            continue
        msub = high_delta[high_delta['method'] == method]
        rhos = []
        for dgp in dgps:
            for delta in msub['delta'].unique():
                dsub = msub[(msub['dgp'] == dgp) & (msub['delta'] == delta)].sort_values('n')
                if len(dsub) >= 3:
                    rho, _ = stats.spearmanr(np.log(dsub['n']), dsub['coverage'])
                    rhos.append(rho)
        if rhos:
            avg_rho = np.mean(rhos)
            t_stat, p_val = stats.ttest_1samp(rhos, 0)
            print(f"  {method:<20}: rho(log_n, cov)={avg_rho:+.4f}, p={p_val:.4f}")

    return method_slopes


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    print("=" * 80)
    print("COMPLETE REANALYSIS")
    print("=" * 80)
    df = load_all()
    print(f"\nLoaded {len(df)} scenarios")
    print(f"DGPs ({len(df['dgp'].unique())}): {sorted(df['dgp'].unique())}")
    print(f"Methods ({len(df['method'].unique())}): {sorted(df['method'].unique())}")
    print(f"Deltas: {sorted(df['delta'].unique())}")
    print(f"Sample sizes: {sorted(df['n'].unique())}")

    n_expected = len(df['dgp'].unique()) * len(df['delta'].unique()) * len(df['n'].unique()) * len(df['method'].unique())
    completeness = len(df) / n_expected * 100 if n_expected > 0 else 0
    print(f"Completeness: {len(df)}/{n_expected} ({completeness:.1f}%)")
    print()

    print("Generating publication figures...")
    fig1_coverage_per_dgp(df, n_val=500)
    fig2_amri_mechanism(df, n_val=500)
    fig3_sample_size(df)
    fig4_heatmap(df)
    fig5_width_tax(df, n_val=500)
    fig6_dgp_comparison(df, n_val=500)

    method_slopes = run_all_tests(df)

    # Save summary CSV
    summary = df.groupby(['method', 'delta']).agg({
        'coverage': ['mean', 'std', 'min', 'max'],
        'avg_width': ['mean', 'std'],
    }).reset_index()
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns]
    summary.to_csv(FIGS / 'FINAL_summary_stats.csv', index=False)
    print(f"\n  Saved: FINAL_summary_stats.csv")

    print(f"\nAll figures saved to: {FIGS}")
    print("REANALYSIS COMPLETE.")
