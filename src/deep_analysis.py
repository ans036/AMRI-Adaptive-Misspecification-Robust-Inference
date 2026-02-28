"""
Deep Analysis: Generate insightful visualizations and form hypotheses
from the intermediate + pilot simulation results.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.family': 'serif',
})

FIGS = Path("c:/Users/anish/OneDrive/Desktop/Novel Research/figures")
FIGS.mkdir(exist_ok=True)

COLORS = {
    'Naive_OLS': '#d62728', 'Sandwich_HC0': '#ff7f0e', 'Sandwich_HC3': '#2ca02c',
    'Pairs_Bootstrap': '#1f77b4', 'Wild_Bootstrap': '#9467bd', 'Bootstrap_t': '#8c564b',
    'Bayesian': '#bcbd22', 'Bayesian_Normal': '#bcbd22', 'AMRI': '#17becf',
}

METHOD_LABELS = {
    'Naive_OLS': 'Naive OLS', 'Sandwich_HC0': 'Sandwich HC0', 'Sandwich_HC3': 'Sandwich HC3',
    'Pairs_Bootstrap': 'Pairs Bootstrap', 'Wild_Bootstrap': 'Wild Bootstrap',
    'Bootstrap_t': 'Bootstrap-t', 'Bayesian': 'Bayesian', 'Bayesian_Normal': 'Bayesian',
    'AMRI': 'AMRI (Proposed)',
}

DGP_LABELS = {
    'DGP1_nonlinearity': 'Nonlinearity\n(Y=X+dX^2+e, fit linear)',
    'DGP3_heteroscedasticity': 'Heteroscedasticity\n(Var(e|X)=exp(dX))',
}

def load_all():
    results_dir = Path("c:/Users/anish/OneDrive/Desktop/Novel Research/results")
    dfs = []
    for f in results_dir.glob("*.csv"):
        dfs.append(pd.read_csv(f))
    df = pd.concat(dfs, ignore_index=True)
    # Normalize column names and method names
    if 'B_valid' in df.columns and 'B' not in df.columns:
        df['B'] = df['B_valid']
    elif 'B_valid' in df.columns:
        df['B'] = df['B'].fillna(df['B_valid'])
    df['method'] = df['method'].replace({'Bayesian_Normal': 'Bayesian'})
    df = df.drop_duplicates(subset=['dgp', 'delta', 'n', 'method'], keep='last')
    return df


# ============================================================================
# FIGURE A: The Central Result - Coverage Degradation Curves
# ============================================================================
def fig_coverage_degradation(df):
    """THE key figure: how coverage degrades with increasing misspecification."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax_idx, n_val in enumerate([100, 500]):
        ax = axes[ax_idx]
        sub = df[df['n'] == n_val]
        dgps = sorted(sub['dgp'].unique())

        # Average across DGPs for each method
        methods_order = ['Naive_OLS', 'Bayesian', 'Sandwich_HC0', 'Sandwich_HC3',
                         'Pairs_Bootstrap', 'Wild_Bootstrap', 'Bootstrap_t', 'AMRI']
        methods_present = [m for m in methods_order if m in sub['method'].unique()]

        for method in methods_present:
            msub = sub[sub['method'] == method]
            avg = msub.groupby('delta')['coverage'].mean().reset_index().sort_values('delta')
            style = '--' if method in ['Naive_OLS', 'Bayesian'] else '-'
            lw = 3 if method == 'AMRI' else 2
            ax.plot(avg['delta'], avg['coverage'], f'o{style}',
                    color=COLORS.get(method, 'black'),
                    label=METHOD_LABELS.get(method, method),
                    linewidth=lw, markersize=6,
                    alpha=0.9 if method == 'AMRI' else 0.7)

        ax.axhline(0.95, color='black', linestyle=':', linewidth=1.5, alpha=0.5)
        ax.axhspan(0.935, 0.965, alpha=0.06, color='green')
        ax.set_xlabel('Misspecification Severity (delta)', fontsize=13)
        ax.set_ylabel('Coverage Probability', fontsize=13)
        ax.set_title(f'n = {n_val}', fontsize=14, fontweight='bold')
        ax.set_ylim(0.55, 1.01)
        ax.grid(True, alpha=0.2)
        ax.legend(loc='lower left', fontsize=9, frameon=True)

    fig.suptitle('Coverage Degradation Under Increasing Misspecification\n'
                 '(Dashed = methods that BREAK; Solid = robust methods; Thick cyan = AMRI)',
                 fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    fig.savefig(FIGS / 'A_coverage_degradation.png', bbox_inches='tight')
    plt.close()
    print("  Saved: A_coverage_degradation.png")


# ============================================================================
# FIGURE B: The Efficiency-Robustness Tradeoff
# ============================================================================
def fig_efficiency_robustness(df):
    """Scatter: coverage vs width showing the fundamental tradeoff."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    deltas_show = [0.0, 0.5, 1.0]
    delta_labels = ['No Misspecification (d=0)', 'Moderate (d=0.5)', 'Severe (d=1.0)']

    for ax_idx, (delta_val, delta_label) in enumerate(zip(deltas_show, delta_labels)):
        ax = axes[ax_idx]
        sub = df[(df['delta'] == delta_val) & (df['n'] == 500)]

        if len(sub) == 0:
            ax.text(0.5, 0.5, 'No data yet', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(delta_label)
            continue

        # Average across DGPs
        agg = sub.groupby('method').agg({'coverage': 'mean', 'avg_width': 'mean'}).reset_index()

        for _, row in agg.iterrows():
            method = row['method']
            marker = '*' if method == 'AMRI' else 'o'
            size = 300 if method == 'AMRI' else 120
            ax.scatter(row['avg_width'], row['coverage'],
                       c=COLORS.get(method, 'black'), s=size, marker=marker,
                       edgecolors='black', linewidth=1, zorder=5)
            offset = (0, 12) if method != 'AMRI' else (0, 15)
            ax.annotate(METHOD_LABELS.get(method, method),
                        (row['avg_width'], row['coverage']),
                        fontsize=8, ha='center', va='bottom',
                        xytext=offset, textcoords='offset points',
                        fontweight='bold' if method == 'AMRI' else 'normal')

        ax.axhline(0.95, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.set_xlabel('Average Interval Width (narrower = more efficient)')
        ax.set_ylabel('Coverage Probability')
        ax.set_title(delta_label, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.2)

    fig.suptitle('Efficiency-Robustness Frontier at n=500\n'
                 'Ideal position: TOP-LEFT (high coverage, narrow intervals)',
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    fig.savefig(FIGS / 'B_efficiency_robustness.png', bbox_inches='tight')
    plt.close()
    print("  Saved: B_efficiency_robustness.png")


# ============================================================================
# FIGURE C: The SE Ratio Diagnostic (Why AMRI works)
# ============================================================================
def fig_se_ratio_mechanism(df):
    """Explain WHY AMRI works: the SE ratio as misspecification detector."""
    sub = df[df['n'] == 500]
    naive_data = sub[sub['method'] == 'Naive_OLS'][['dgp', 'delta', 'avg_width']].rename(
        columns={'avg_width': 'w_naive'})
    hc3_data = sub[sub['method'] == 'Sandwich_HC3'][['dgp', 'delta', 'avg_width']].rename(
        columns={'avg_width': 'w_hc3'})

    merged = naive_data.merge(hc3_data, on=['dgp', 'delta'])
    merged['se_ratio'] = merged['w_hc3'] / merged['w_naive']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: SE ratio vs delta
    for dgp in sorted(merged['dgp'].unique()):
        dsub = merged[merged['dgp'] == dgp].sort_values('delta')
        label = dgp.split('_', 1)[1].replace('_', ' ').title()
        ax1.plot(dsub['delta'], dsub['se_ratio'], 'o-', linewidth=2, markersize=7, label=label)

    ax1.axhline(1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    # Show AMRI threshold zone
    for n_val in [100, 500]:
        thresh = 1 + 2 / np.sqrt(n_val)
        ax1.axhline(thresh, color='gray', linestyle=':', alpha=0.3)
        ax1.text(1.02, thresh, f'n={n_val}\nthreshold', fontsize=8, va='center')

    ax1.set_xlabel('Misspecification Severity (delta)', fontsize=12)
    ax1.set_ylabel('SE Ratio (Sandwich HC3 / Naive)', fontsize=12)
    ax1.set_title('Misspecification Diagnostic:\nSandwich-to-Naive SE Ratio', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.2)

    # Right: Coverage of naive vs sandwich vs AMRI
    methods_show = ['Naive_OLS', 'Sandwich_HC3', 'AMRI']
    for method in methods_show:
        msub = sub[sub['method'] == method]
        avg = msub.groupby('delta')['coverage'].mean().reset_index().sort_values('delta')
        style = '--' if method == 'Naive_OLS' else '-'
        lw = 3 if method == 'AMRI' else 2
        ax2.plot(avg['delta'], avg['coverage'], f'o{style}',
                 color=COLORS.get(method, 'black'),
                 label=METHOD_LABELS.get(method, method),
                 linewidth=lw, markersize=7)

    ax2.axhline(0.95, color='black', linestyle=':', linewidth=1)
    ax2.set_xlabel('Misspecification Severity (delta)', fontsize=12)
    ax2.set_ylabel('Coverage Probability', fontsize=12)
    ax2.set_title('AMRI Switches Between Naive & Sandwich\nBased on Detected Misspecification', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.set_ylim(0.55, 1.01)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(FIGS / 'C_amri_mechanism.png', bbox_inches='tight')
    plt.close()
    print("  Saved: C_amri_mechanism.png")


# ============================================================================
# FIGURE D: Coverage vs Sample Size
# ============================================================================
def fig_coverage_vs_n(df):
    """How does coverage change with sample size under misspecification?"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax_idx, delta_val in enumerate([0.5, 1.0]):
        ax = axes[ax_idx]
        sub = df[df['delta'] == delta_val]
        if len(sub) == 0:
            ax.set_title(f'delta={delta_val} (no data yet)')
            continue

        methods_show = ['Naive_OLS', 'Sandwich_HC3', 'Pairs_Bootstrap', 'Bootstrap_t', 'AMRI']
        for method in methods_show:
            if method not in sub['method'].unique():
                continue
            msub = sub[sub['method'] == method]
            avg = msub.groupby('n')['coverage'].mean().reset_index().sort_values('n')
            style = '--' if method == 'Naive_OLS' else '-'
            lw = 3 if method == 'AMRI' else 2
            ax.plot(avg['n'], avg['coverage'], f'o{style}',
                    color=COLORS.get(method, 'black'),
                    label=METHOD_LABELS.get(method, method),
                    linewidth=lw, markersize=6)

        ax.axhline(0.95, color='black', linestyle=':', linewidth=1.5, alpha=0.5)
        ax.set_xlabel('Sample Size (n)', fontsize=12)
        ax.set_ylabel('Coverage Probability', fontsize=12)
        ax.set_title(f'Misspecification Severity delta={delta_val}', fontsize=13, fontweight='bold')
        ax.set_xscale('log')
        ax.set_ylim(0.5, 1.01)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.2)

    fig.suptitle('Coverage vs Sample Size: More Data WORSENS Naive Inference Under Misspecification',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGS / 'D_coverage_vs_n.png', bbox_inches='tight')
    plt.close()
    print("  Saved: D_coverage_vs_n.png")


# ============================================================================
# FIGURE E: Heatmap
# ============================================================================
def fig_heatmap(df):
    """Coverage heatmap: methods x (delta, n)."""
    sub = df.copy()
    sub['scenario'] = 'd=' + sub['delta'].astype(str) + '\nn=' + sub['n'].astype(str)

    # Average across DGPs
    agg = sub.groupby(['method', 'delta', 'n'])['coverage'].mean().reset_index()
    agg['scenario'] = 'd=' + agg['delta'].astype(str) + ', n=' + agg['n'].astype(str)
    pivot = agg.pivot_table(values='coverage', index='method', columns='scenario')

    # Sort columns
    cols_order = sorted(pivot.columns, key=lambda x: (float(x.split(',')[0].split('=')[1]),
                                                        float(x.split(',')[1].split('=')[1])))
    pivot = pivot[cols_order]

    fig, ax = plt.subplots(figsize=(20, 7))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0.95,
                vmin=0.5, vmax=1.0, ax=ax, linewidths=0.5,
                cbar_kws={'label': 'Coverage Probability'})
    ax.set_title('Coverage Heatmap: Methods x Scenarios (nominal = 0.95)\n'
                 'Green = valid coverage | Red = coverage failure',
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Method', fontsize=12)
    ax.set_xlabel('Scenario (delta, n)', fontsize=12)
    plt.tight_layout()
    fig.savefig(FIGS / 'E_coverage_heatmap.png', bbox_inches='tight')
    plt.close()
    print("  Saved: E_coverage_heatmap.png")


# ============================================================================
# FIGURE F: The "Width Tax" of Robustness
# ============================================================================
def fig_width_tax(df):
    """Show the efficiency cost of using robust methods."""
    sub = df[df['n'] == 500]
    if len(sub) == 0:
        return

    # Compute width relative to naive
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
    methods_show = ['Sandwich_HC3', 'Pairs_Bootstrap', 'Bootstrap_t', 'AMRI']
    for method in methods_show:
        msub = all_ratios[all_ratios['method'] == method]
        avg = msub.groupby('delta')['width_ratio'].mean().reset_index().sort_values('delta')
        lw = 3 if method == 'AMRI' else 2
        ax.plot(avg['delta'], avg['width_ratio'], 'o-',
                color=COLORS.get(method, 'black'),
                label=METHOD_LABELS.get(method, method),
                linewidth=lw, markersize=7)

    ax.axhline(1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.set_xlabel('Misspecification Severity (delta)', fontsize=13)
    ax.set_ylabel('Width Ratio (Method / Naive OLS)', fontsize=13)
    ax.set_title('The "Robustness Tax": How Much Wider Are Robust Intervals?\n'
                 'Ratio = 1 means same width as naive; >1 means wider (less efficient)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig(FIGS / 'F_width_tax.png', bbox_inches='tight')
    plt.close()
    print("  Saved: F_width_tax.png")


# ============================================================================
# HYPOTHESIS FORMULATION
# ============================================================================
def formulate_hypotheses(df):
    """Analyze data patterns and formulate formal hypotheses."""
    print("\n" + "=" * 80)
    print("HYPOTHESIS FORMULATION FROM EMPIRICAL EVIDENCE")
    print("=" * 80)

    sub500 = df[df['n'] == 500]

    # H1: Naive methods fail monotonically
    print("\n--- HYPOTHESIS H1: Monotonic Coverage Degradation ---")
    for method in ['Naive_OLS', 'Bayesian']:
        if method not in sub500['method'].values:
            continue
        msub = sub500[sub500['method'] == method]
        avg = msub.groupby('delta')['coverage'].mean().sort_index()
        diffs = avg.diff().dropna()
        all_decreasing = (diffs <= 0.005).all()  # allow tiny noise
        print(f"  {method}: coverages = {dict(zip(avg.index, avg.values.round(3)))}")
        print(f"    Monotonically decreasing? {all_decreasing}")

    # H2: Sandwich methods maintain coverage
    print("\n--- HYPOTHESIS H2: Sandwich SE Maintains Nominal Coverage ---")
    for method in ['Sandwich_HC3', 'Sandwich_HC0']:
        if method not in sub500['method'].values:
            continue
        msub = sub500[sub500['method'] == method]
        avg = msub.groupby('delta')['coverage'].mean()
        min_cov = avg.min()
        max_cov = avg.max()
        print(f"  {method}: min_coverage={min_cov:.4f}, max_coverage={max_cov:.4f}")
        print(f"    Always >= 0.93? {min_cov >= 0.93}")

    # H3: AMRI is best of both worlds
    print("\n--- HYPOTHESIS H3: AMRI Achieves Best of Both Worlds ---")
    if 'AMRI' in sub500['method'].values:
        amri = sub500[sub500['method'] == 'AMRI']
        amri_avg = amri.groupby('delta')['coverage'].mean()
        naive = sub500[sub500['method'] == 'Naive_OLS']
        naive_avg = naive.groupby('delta')['coverage'].mean()

        # At delta=0: AMRI should be close to naive (efficient)
        deltas_present = sorted(amri_avg.index)
        if 0.0 in deltas_present:
            print(f"  At d=0.0: AMRI={amri_avg[0.0]:.4f}, Naive={naive_avg[0.0]:.4f}")
            print(f"    AMRI matches naive efficiency? {abs(amri_avg[0.0] - naive_avg[0.0]) < 0.02}")

        # At delta=1: AMRI should be close to sandwich (robust)
        max_delta = max(deltas_present)
        if 'Sandwich_HC3' in sub500['method'].values:
            hc3 = sub500[sub500['method'] == 'Sandwich_HC3']
            hc3_avg = hc3.groupby('delta')['coverage'].mean()
            if max_delta in hc3_avg.index:
                print(f"  At d={max_delta}: AMRI={amri_avg[max_delta]:.4f}, HC3={hc3_avg[max_delta]:.4f}")
                print(f"    AMRI >= HC3? {amri_avg[max_delta] >= hc3_avg[max_delta] - 0.01}")

    # H4: Coverage degradation rate
    print("\n--- HYPOTHESIS H4: Coverage Degradation Rate ---")
    for method in sorted(sub500['method'].unique()):
        msub = sub500[sub500['method'] == method]
        avg = msub.groupby('delta')['coverage'].mean()
        deltas_present = sorted(avg.index)
        if len(deltas_present) >= 2:
            d_min, d_max = deltas_present[0], deltas_present[-1]
            drop = avg[d_min] - avg[d_max]
            rate = drop / (d_max - d_min) if d_max > d_min else 0
            print(f"  {method:<20}: total_drop={drop:+.4f}, rate={rate:.4f}/unit_delta")

    # H5: Width grows with misspecification for robust methods
    print("\n--- HYPOTHESIS H5: Width Grows with Severity for Robust Methods ---")
    for method in ['Sandwich_HC3', 'AMRI', 'Pairs_Bootstrap']:
        if method not in sub500['method'].values:
            continue
        msub = sub500[sub500['method'] == method]
        avg_w = msub.groupby('delta')['avg_width'].mean()
        widths_str = ', '.join([f'd={d}:{w:.4f}' for d, w in avg_w.items()])
        is_monotone = all(avg_w.diff().dropna() >= -0.001)
        print(f"  {method}: {widths_str}")
        print(f"    Monotonically increasing? {is_monotone}")

    # Summary of formal hypotheses
    print("\n" + "=" * 80)
    print("FORMAL HYPOTHESES")
    print("=" * 80)
    print("""
    H1 (Misspecification Harms Naive Inference):
        Under model misspecification with severity delta > 0,
        the coverage of model-based (naive) confidence intervals
        monotonically decreases as delta increases, and can fall
        arbitrarily far below the nominal level.

    H2 (Sandwich Robustness):
        Sandwich standard error-based confidence intervals (HC3)
        maintain coverage probability >= 0.93 for all delta in [0,1]
        and all sample sizes n >= 50, regardless of misspecification type.

    H3 (AMRI Best-of-Both-Worlds):
        The AMRI method achieves:
        (a) Coverage within 1% of naive when delta=0 (efficiency)
        (b) Coverage >= sandwich HC3 when delta>0 (robustness)
        (c) Interval width <= 1.1x sandwich HC3 width (bounded tax)

    H4 (Degradation Rate Ordering):
        The coverage degradation rate satisfies:
        Naive >> Wild_Bootstrap > Pairs_Bootstrap >= Sandwich_HC0
        > Sandwich_HC3 >= Bootstrap_t >= AMRI

    H5 (Adaptive Width):
        Robust methods (sandwich, bootstrap, AMRI) automatically
        widen their intervals in proportion to the degree of
        misspecification, while naive methods maintain constant
        width regardless of misspecification severity.
    """)


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    print("Loading all available results...")
    df = load_all()
    print(f"Total: {len(df)} scenarios")
    print(f"DGPs: {sorted(df.dgp.unique())}")
    print(f"Methods: {sorted(df.method.unique())}")
    print(f"Deltas: {sorted(df.delta.unique())}")
    print(f"Sample sizes: {sorted(df.n.unique())}")
    print()

    print("Generating visualizations...")
    fig_coverage_degradation(df)
    fig_efficiency_robustness(df)
    fig_se_ratio_mechanism(df)
    fig_coverage_vs_n(df)
    fig_heatmap(df)
    fig_width_tax(df)

    print("\nFormulating hypotheses...")
    formulate_hypotheses(df)

    print(f"\nAll figures saved to: {FIGS}")
    print("DONE.")
