"""
Publication-Quality Visualizations for AMRI v2
==============================================

Generates 6 figures capturing the key insights of the AMRI research:

  Figure 1: "The AMRI Story" -- convergence rate proof (QQ + tau stabilization)
  Figure 2: Threshold Sensitivity Heatmap -- (c1, c2) landscape
  Figure 3: Blending Weight Adaptation -- how w responds to misspecification
  Figure 4: Generalized Coverage Comparison -- AMRI across estimator classes
  Figure 5: The Coverage-Width Tradeoff -- Pareto frontier across all methods
  Figure 6: Head-to-Head Dashboard -- v1 vs v2 across all DGPs

All figures saved to figures/ as both PNG (300 dpi) and PDF.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# Style
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.family': 'serif',
})

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# Color palette
C_NAIVE = '#e74c3c'     # red
C_ROBUST = '#3498db'    # blue
C_AMRI = '#2ecc71'      # green
C_AMRI_V1 = '#f39c12'   # orange
C_AMRI_V2 = '#2ecc71'   # green
C_HC3 = '#3498db'       # blue
C_ACCENT = '#9b59b6'    # purple
C_GRAY = '#95a5a6'


def save_fig(fig, name):
    """Save figure as PNG and PDF."""
    fig.savefig(os.path.join(FIG_DIR, f'{name}.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(FIG_DIR, f'{name}.pdf'), bbox_inches='tight')
    print(f"  Saved {name}.png and {name}.pdf")


# =========================================================================
# FIGURE 1: The Convergence Rate Proof
# =========================================================================

def figure1_convergence_proof():
    """Show that sqrt(n)*log(R) -> N(0,1) and tau stabilizes at 1.0."""
    print("Figure 1: Convergence Rate Proof...")

    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # --- Panel A: Distribution of sqrt(n)*log(R) for several n ---
    ax = axes[0]
    n_values = [50, 200, 1000]
    colors = ['#e74c3c', '#f39c12', '#2ecc71']

    for n_val, color in zip(n_values, colors):
        B = min(50000, 200_000_000 // (n_val * 8))
        X = rng.standard_normal((B, n_val))
        eps = rng.standard_normal((B, n_val))
        Y = 1.0 + X + eps

        # OLS
        Xbar = X.mean(axis=1, keepdims=True)
        Ybar = Y.mean(axis=1, keepdims=True)
        Xc = X - Xbar
        Yc = Y - Ybar
        SXX = (Xc**2).sum(axis=1)
        SXY = (Xc * Yc).sum(axis=1)
        slopes = SXY / SXX
        intercepts = Ybar.squeeze() - slopes * Xbar.squeeze()
        fitted = intercepts[:, None] + slopes[:, None] * X
        resid = Y - fitted
        sigma2 = (resid**2).sum(axis=1) / (n_val - 2)
        se_naive = np.sqrt(sigma2 / SXX)
        h = 1.0/n_val + Xc**2 / SXX[:, None]
        adj_resid = resid / (1.0 - h)
        meat = (Xc**2 * adj_resid**2).sum(axis=1)
        se_hc3 = np.sqrt(meat / SXX**2)

        scaled = np.sqrt(n_val) * np.log(se_hc3 / se_naive)
        scaled = scaled[np.isfinite(scaled)]

        ax.hist(scaled, bins=80, density=True, alpha=0.45, color=color,
                label=f'n={n_val}', edgecolor='white', linewidth=0.3)

    # Overlay N(0,1)
    x_grid = np.linspace(-4, 4, 200)
    ax.plot(x_grid, stats.norm.pdf(x_grid), 'k-', lw=2.0, label='N(0, 1)',
            zorder=10)
    ax.set_xlabel(r'$\sqrt{n} \cdot \log(R_n)$')
    ax.set_ylabel('Density')
    ax.set_title('(A)  CLT for the SE Ratio')
    ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc')
    ax.set_xlim(-4, 4)

    # --- Panel B: tau stabilization ---
    ax = axes[1]
    df_conv = pd.read_csv(os.path.join(RESULTS_DIR, 'results_convergence_rate.csv'))
    ax.plot(df_conv['n'], df_conv['tau'], 'o-', color=C_AMRI_V2, markersize=8,
            linewidth=2, markeredgecolor='white', markeredgewidth=1.5, zorder=5)
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, linewidth=1)
    ax.fill_between(df_conv['n'], 0.95, 1.05, alpha=0.1, color=C_AMRI_V2)
    ax.set_xlabel('Sample size n')
    ax.set_ylabel(r'Estimated $\tau$')
    ax.set_title(r'(B)  $\tau$ Stabilizes at 1.0')
    ax.set_xscale('log')
    ax.set_ylim(0.9, 1.1)
    ax.set_xticks([50, 100, 250, 500, 1000, 5000])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    # Annotate
    for _, row in df_conv.iterrows():
        ax.annotate(f'{row["tau"]:.3f}',
                    (row['n'], row['tau']),
                    textcoords="offset points", xytext=(0, 12),
                    ha='center', fontsize=8, color='#555555')

    # --- Panel C: 1/sqrt(n) scaling verification ---
    ax = axes[2]
    ax.plot(df_conv['n'], df_conv['p95_scaled'], 's-', color=C_ROBUST,
            markersize=7, linewidth=2, label='95th pctile $\\times \\sqrt{n}$',
            markeredgecolor='white', markeredgewidth=1)
    ax.plot(df_conv['n'], df_conv['p99_scaled'], '^-', color=C_NAIVE,
            markersize=7, linewidth=2, label='99th pctile $\\times \\sqrt{n}$',
            markeredgecolor='white', markeredgewidth=1)
    ax.axhline(y=1.96, color=C_ROBUST, linestyle=':', alpha=0.5)
    ax.axhline(y=2.576, color=C_NAIVE, linestyle=':', alpha=0.5)
    ax.set_xlabel('Sample size n')
    ax.set_ylabel(r'Percentile $\times \sqrt{n}$')
    ax.set_title('(C)  1/$\\sqrt{n}$ Scaling Confirmed')
    ax.set_xscale('log')
    ax.set_xticks([50, 100, 250, 500, 1000, 5000])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc')
    ax.set_ylim(1.5, 3.0)

    fig.suptitle('Formal Threshold Derivation: Why $c/\\sqrt{n}$ Scaling',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig(fig, 'FIG_threshold_proof')
    plt.close(fig)


# =========================================================================
# FIGURE 2: Threshold Sensitivity Heatmap
# =========================================================================

def figure2_threshold_heatmap():
    """2D heatmap showing coverage and regret as function of (c1, c2)."""
    print("Figure 2: Threshold Sensitivity Heatmap...")

    df = pd.read_csv(os.path.join(RESULTS_DIR, 'results_threshold_proof.csv'))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for idx, (metric, title, cmap_name, vmin, vmax) in enumerate([
        ('mean_coverage', 'Mean Coverage', 'RdYlGn', 0.944, 0.951),
        ('max_regret', 'Max Regret (lower = better)', 'RdYlGn_r', None, None),
    ]):
        ax = axes[idx]
        pivot = df.pivot_table(index='c1', columns='c2', values=metric)

        if vmin is None:
            vmin = pivot.min().min()
            vmax = pivot.max().max()

        im = ax.imshow(pivot.values, aspect='auto', cmap=cmap_name,
                       vmin=vmin, vmax=vmax, origin='lower',
                       extent=[pivot.columns.min()-0.125, pivot.columns.max()+0.125,
                               pivot.index.min()-0.125, pivot.index.max()+0.125])

        # Annotate cells
        for i, c1 in enumerate(pivot.index):
            for j, c2 in enumerate(pivot.columns):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    if metric == 'mean_coverage':
                        txt = f'{val:.4f}'
                    else:
                        txt = f'{val:.1e}'
                    text_color = 'white' if (val < (vmin + vmax)/2 and metric == 'mean_coverage') or \
                                            (val > (vmin + vmax)/2 and metric == 'max_regret') else 'black'
                    ax.text(c2, c1, txt, ha='center', va='center',
                            fontsize=8, color=text_color, fontweight='bold')

        # Mark heuristic (1.0, 2.0)
        ax.plot(2.0, 1.0, '*', color='red', markersize=18, markeredgecolor='white',
                markeredgewidth=1.5, zorder=10)
        ax.annotate('Heuristic\n(1.0, 2.0)', (2.0, 1.0),
                    textcoords="offset points", xytext=(25, -20),
                    fontsize=9, color='red', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

        # Mark optimal
        if metric == 'max_regret':
            best = df.loc[df['max_regret'].idxmin()]
            ax.plot(best['c2'], best['c1'], 'D', color='blue', markersize=10,
                    markeredgecolor='white', markeredgewidth=1.5, zorder=10)
            ax.annotate(f'Optimal\n({best["c1"]:.1f}, {best["c2"]:.1f})',
                        (best['c2'], best['c1']),
                        textcoords="offset points", xytext=(-35, 15),
                        fontsize=9, color='blue', fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))

        ax.set_xlabel('$c_2$ (full-robust threshold)', fontsize=11)
        ax.set_ylabel('$c_1$ (start-blending threshold)', fontsize=11)
        ax.set_title(f'({chr(65+idx)})  {title}', fontsize=12)
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle('Threshold Sensitivity: Performance is Flat Near (1.0, 2.0)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig(fig, 'FIG_threshold_heatmap')
    plt.close(fig)


# =========================================================================
# FIGURE 3: Blending Weight Adaptation
# =========================================================================

def figure3_blending_weights():
    """Show how the blending weight w adapts to misspecification severity."""
    print("Figure 3: Blending Weight Adaptation...")

    df_gen = pd.read_csv(os.path.join(RESULTS_DIR, 'results_generalized.csv'))

    # Select key DGPs that show the adaptation story
    dgp_stories = {
        'MLR_hetero': ('MLR: Heteroscedasticity', 'Variance misspec detected'),
        'MLR_correct': ('MLR: Correct Spec', 'No misspec, stays efficient'),
        'Poisson_overdispersed': ('Poisson: Overdispersion', 'Overdispersion detected'),
        'Logistic_link': ('Logistic: Link Misspec', 'Bias not detected (expected)'),
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    for idx, (dgp_name, (display_name, subtitle)) in enumerate(dgp_stories.items()):
        ax = axes[idx]
        sub = df_gen[df_gen['dgp'] == dgp_name].copy()

        for n_val, marker, color in [(100, 'o', '#e74c3c'), (300, 's', '#f39c12'),
                                      (1000, '^', '#2ecc71')]:
            nsub = sub[sub['n'] == n_val].sort_values('delta')
            if len(nsub) == 0:
                continue

            # Plot blending weight
            ax.plot(nsub['delta'], nsub['avg_weight'], f'{marker}-',
                    color=color, markersize=7, linewidth=2,
                    label=f'n={n_val}', markeredgecolor='white',
                    markeredgewidth=1)

        ax.axhline(y=0.0, color='gray', linestyle=':', alpha=0.3)
        ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.3)
        ax.fill_between([0, 1], 0, 0.15, alpha=0.08, color=C_NAIVE,
                        label='_nolegend_')
        ax.fill_between([0, 1], 0.85, 1.0, alpha=0.08, color=C_ROBUST,
                        label='_nolegend_')

        ax.text(0.02, 0.05, 'Efficient mode', fontsize=8, color=C_NAIVE,
                alpha=0.7, transform=ax.transAxes)
        ax.text(0.02, 0.92, 'Robust mode', fontsize=8, color=C_ROBUST,
                alpha=0.7, transform=ax.transAxes)

        ax.set_xlabel('Misspecification severity $\\delta$')
        ax.set_ylabel('Blending weight $w$')
        ax.set_title(f'({chr(65+idx)})  {display_name}', fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(-0.02, 1.02)
        ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc',
                  loc='center right')

        # Add subtitle annotation
        ax.text(0.98, 0.15, subtitle, transform=ax.transAxes, fontsize=9,
                ha='right', style='italic', color='#555555',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                          edgecolor='#dddddd', alpha=0.9))

    fig.suptitle('AMRI v2 Blending Weight: Intelligent Adaptation to Misspecification',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_fig(fig, 'FIG_blending_weights')
    plt.close(fig)


# =========================================================================
# FIGURE 4: Generalized Coverage Comparison
# =========================================================================

def figure4_generalized_coverage():
    """Coverage comparison across 3 estimator classes."""
    print("Figure 4: Generalized Coverage Comparison...")

    df_gen = pd.read_csv(os.path.join(RESULTS_DIR, 'results_generalized.csv'))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    categories = {
        'MLR': ('Multiple Linear Regression (p=3)',
                ['MLR_correct', 'MLR_hetero', 'MLR_omitted']),
        'Logistic': ('Logistic Regression',
                     ['Logistic_correct', 'Logistic_hetero']),
        'Poisson': ('Poisson Regression',
                    ['Poisson_correct', 'Poisson_overdispersed', 'Poisson_zero_infl']),
    }

    for idx, (cat, (title, dgps)) in enumerate(categories.items()):
        ax = axes[idx]
        sub = df_gen[(df_gen['category'] == cat) & (df_gen['dgp'].isin(dgps))]
        # Exclude the ones where all methods fail (logistic link misspec)
        sub = sub[sub['cov_naive'] > 0.3]

        # Aggregate by delta (mean across DGPs within category and n)
        agg = sub.groupby('delta').agg({
            'cov_naive': 'mean',
            'cov_robust': 'mean',
            'cov_amri': 'mean',
        }).reset_index()

        ax.plot(agg['delta'], agg['cov_naive'], 'o-', color=C_NAIVE,
                linewidth=2.5, markersize=8, label='Naive (model-based)',
                markeredgecolor='white', markeredgewidth=1.5)
        ax.plot(agg['delta'], agg['cov_robust'], 's-', color=C_ROBUST,
                linewidth=2.5, markersize=8, label='Robust (sandwich)',
                markeredgecolor='white', markeredgewidth=1.5)
        ax.plot(agg['delta'], agg['cov_amri'], '^-', color=C_AMRI_V2,
                linewidth=2.5, markersize=9, label='AMRI v2',
                markeredgecolor='white', markeredgewidth=1.5)

        ax.axhline(y=0.95, color='black', linestyle='--', alpha=0.4, linewidth=1)
        ax.fill_between(agg['delta'], 0.94, 0.96, alpha=0.06, color='gray')

        ax.set_xlabel('Misspecification severity $\\delta$')
        ax.set_ylabel('Coverage')
        ax.set_title(f'({chr(65+idx)})  {title}', fontsize=12)
        ax.set_ylim(0.83, 0.98)
        ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc',
                  loc='lower left')
        ax.set_xlim(-0.02, 1.02)

    fig.suptitle('AMRI v2 Generalizes: Coverage Across Estimator Classes',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig(fig, 'FIG_generalized_coverage')
    plt.close(fig)


# =========================================================================
# FIGURE 5: The Efficiency-Robustness Tradeoff
# =========================================================================

def figure5_efficiency_robustness():
    """Coverage vs Width tradeoff showing AMRI dominates the Pareto frontier."""
    print("Figure 5: Efficiency-Robustness Tradeoff...")

    df_gen = pd.read_csv(os.path.join(RESULTS_DIR, 'results_generalized.csv'))
    # Focus on MLR and Poisson (where there's a real tradeoff)
    df_sub = df_gen[df_gen['category'].isin(['MLR', 'Poisson'])]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Panel A: Per-scenario scatter ---
    ax = axes[0]
    ax.scatter(df_sub['width_naive'], df_sub['cov_naive'],
               c=C_NAIVE, alpha=0.4, s=30, label='Naive', edgecolors='none')
    ax.scatter(df_sub['width_robust'], df_sub['cov_robust'],
               c=C_ROBUST, alpha=0.4, s=30, label='Robust', edgecolors='none')
    ax.scatter(df_sub['width_amri'], df_sub['cov_amri'],
               c=C_AMRI_V2, alpha=0.5, s=40, label='AMRI v2',
               edgecolors='white', linewidths=0.5, zorder=5)

    ax.axhline(y=0.95, color='black', linestyle='--', alpha=0.4, linewidth=1)
    ax.set_xlabel('Average CI Width (narrower is better)')
    ax.set_ylabel('Coverage (higher is better)')
    ax.set_title('(A)  Per-Scenario Coverage vs Width')
    ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc')

    # Add annotation arrow showing AMRI's "sweet spot"
    ax.annotate('AMRI: narrow + high coverage',
                xy=(df_sub['width_amri'].median(), df_sub['cov_amri'].median()),
                xytext=(0.42, 0.92),
                fontsize=9, color=C_AMRI_V2, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_AMRI_V2, lw=1.5))

    # --- Panel B: Aggregated bar chart ---
    ax = axes[1]
    cats = ['MLR', 'Poisson']
    x = np.arange(len(cats))
    width = 0.22

    for i, (method, col_cov, col_w, color, label) in enumerate([
        ('Naive', 'cov_naive', 'width_naive', C_NAIVE, 'Naive'),
        ('Robust', 'cov_robust', 'width_robust', C_ROBUST, 'Robust'),
        ('AMRI', 'cov_amri', 'width_amri', C_AMRI_V2, 'AMRI v2'),
    ]):
        covs = []
        widths = []
        for cat in cats:
            sub = df_sub[df_sub['category'] == cat]
            covs.append(sub[col_cov].mean())
            widths.append(sub[col_w].mean())

        bars = ax.bar(x + (i - 1) * width, covs, width, color=color,
                      label=label, edgecolor='white', linewidth=0.5,
                      alpha=0.85)

        # Annotate with width
        for j, (bar, w) in enumerate(zip(bars, widths)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f'w={w:.3f}', ha='center', va='bottom', fontsize=7.5,
                    color='#555555', rotation=0)

    ax.axhline(y=0.95, color='black', linestyle='--', alpha=0.4, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=11)
    ax.set_ylabel('Mean Coverage')
    ax.set_title('(B)  Coverage by Estimator Class (width annotated)')
    ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc')
    ax.set_ylim(0.85, 0.98)

    fig.suptitle('The Efficiency-Robustness Tradeoff: AMRI Achieves Both',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig(fig, 'FIG_efficiency_robustness')
    plt.close(fig)


# =========================================================================
# FIGURE 6: v1 vs v2 Head-to-Head Dashboard
# =========================================================================

def figure6_v1_vs_v2():
    """Comprehensive v1 vs v2 comparison from standalone simulation."""
    print("Figure 6: v1 vs v2 Head-to-Head Dashboard...")

    df = pd.read_csv(os.path.join(RESULTS_DIR, 'results_amri_v2.csv'))
    df_v1 = df[df['method'] == 'AMRI']
    df_v2 = df[df['method'] == 'AMRI_v2']

    # Merge v1 and v2
    merge_keys = ['dgp', 'delta', 'n']
    merged = pd.merge(df_v1, df_v2, on=merge_keys, suffixes=('_v1', '_v2'))

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)

    # --- Panel A: Coverage v1 vs v2 scatter ---
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(merged['coverage_v1'], merged['coverage_v2'],
               c=merged['delta'], cmap='RdYlGn_r', s=35, alpha=0.7,
               edgecolors='white', linewidths=0.5, zorder=5)
    lims = [0.88, 1.0]
    ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=1)
    ax.set_xlabel('AMRI v1 Coverage')
    ax.set_ylabel('AMRI v2 Coverage')
    ax.set_title('(A)  Coverage: v1 vs v2')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')

    # Color bar
    sm = plt.cm.ScalarMappable(cmap='RdYlGn_r',
                                norm=plt.Normalize(0, 1))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, shrink=0.8)
    cb.set_label('$\\delta$', fontsize=10)

    # --- Panel B: Width v1 vs v2 scatter ---
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(merged['avg_width_v1'], merged['avg_width_v2'],
               c=C_AMRI_V2, s=35, alpha=0.5, edgecolors='white',
               linewidths=0.5)
    w_lims = [merged[['avg_width_v1', 'avg_width_v2']].min().min() * 0.95,
              merged[['avg_width_v1', 'avg_width_v2']].max().max() * 1.05]
    ax.plot(w_lims, w_lims, 'k--', alpha=0.3, linewidth=1)
    ax.set_xlabel('AMRI v1 Width')
    ax.set_ylabel('AMRI v2 Width')
    ax.set_title('(B)  Width: v2 is Narrower')
    pct_narrower = (merged['avg_width_v2'] < merged['avg_width_v1']).mean()
    ax.text(0.05, 0.92, f'v2 narrower in {100*pct_narrower:.0f}%\nof scenarios',
            transform=ax.transAxes, fontsize=10, color=C_AMRI_V2,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                      edgecolor='#dddddd'))

    # --- Panel C: Coverage by DGP ---
    ax = fig.add_subplot(gs[0, 2])
    dgps = merged['dgp'].unique()
    dgp_labels = [d.replace('DGP', '').replace('_', '\n') for d in dgps]
    x = np.arange(len(dgps))
    width = 0.35

    cov_v1 = [merged[merged['dgp'] == d]['coverage_v1'].mean() for d in dgps]
    cov_v2 = [merged[merged['dgp'] == d]['coverage_v2'].mean() for d in dgps]

    bars1 = ax.bar(x - width/2, cov_v1, width, color=C_AMRI_V1, label='v1',
                   alpha=0.85, edgecolor='white')
    bars2 = ax.bar(x + width/2, cov_v2, width, color=C_AMRI_V2, label='v2',
                   alpha=0.85, edgecolor='white')
    ax.axhline(y=0.95, color='black', linestyle='--', alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(dgp_labels, fontsize=7.5)
    ax.set_ylabel('Mean Coverage')
    ax.set_title('(C)  Coverage by DGP')
    ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc')
    ax.set_ylim(0.92, 0.97)

    # --- Panel D: Coverage variability (std) ---
    ax = fig.add_subplot(gs[1, 0])
    std_v1 = [merged[merged['dgp'] == d]['coverage_v1'].std() for d in dgps]
    std_v2 = [merged[merged['dgp'] == d]['coverage_v2'].std() for d in dgps]
    ax.bar(x - width/2, std_v1, width, color=C_AMRI_V1, label='v1',
           alpha=0.85, edgecolor='white')
    ax.bar(x + width/2, std_v2, width, color=C_AMRI_V2, label='v2',
           alpha=0.85, edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(dgp_labels, fontsize=7.5)
    ax.set_ylabel('Coverage Std Dev')
    ax.set_title('(D)  Uniformity: v2 is Less Variable')
    ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc')

    # --- Panel E: Blending weight distribution ---
    ax = fig.add_subplot(gs[1, 1])
    v2_data = df_v2.dropna(subset=['avg_weight'])
    for delta_val, color, marker in [(0.0, '#2ecc71', 'o'),
                                      (0.5, '#f39c12', 's'),
                                      (1.0, '#e74c3c', '^')]:
        dsub = v2_data[v2_data['delta'] == delta_val]
        if len(dsub) > 0:
            agg_w = dsub.groupby('n')['avg_weight'].mean().reset_index().sort_values('n')
            ax.plot(agg_w['n'], agg_w['avg_weight'],
                    f'{marker}-', color=color, markersize=7, linewidth=2,
                    label=f'$\\delta$={delta_val}',
                    markeredgecolor='white', markeredgewidth=1)

    ax.set_xlabel('Sample size n')
    ax.set_ylabel('Average blending weight $w$')
    ax.set_title('(E)  Weight Adapts with n and $\\delta$')
    ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc')
    ax.set_xscale('log')

    # --- Panel F: Summary stats box ---
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')

    stats_text = [
        ('AMRI v2 vs v1 Summary', '', True),
        ('', '', False),
        ('Mean coverage v1:', f'{merged["coverage_v1"].mean():.4f}', False),
        ('Mean coverage v2:', f'{merged["coverage_v2"].mean():.4f}', False),
        ('Coverage std v1:', f'{merged["coverage_v1"].std():.4f}', False),
        ('Coverage std v2:', f'{merged["coverage_v2"].std():.4f}', False),
        ('v2 narrower:', f'{100*pct_narrower:.1f}% of scenarios', False),
        ('Min coverage v1:', f'{merged["coverage_v1"].min():.4f}', False),
        ('Min coverage v2:', f'{merged["coverage_v2"].min():.4f}', False),
    ]

    y_pos = 0.92
    for label, value, is_header in stats_text:
        if is_header:
            ax.text(0.5, y_pos, label, transform=ax.transAxes,
                    fontsize=13, ha='center', fontweight='bold',
                    color='#333333')
        elif label:
            ax.text(0.15, y_pos, label, transform=ax.transAxes,
                    fontsize=10, ha='left', color='#555555')
            ax.text(0.85, y_pos, value, transform=ax.transAxes,
                    fontsize=10, ha='right', fontweight='bold',
                    color=C_AMRI_V2)
        y_pos -= 0.10

    # Box around stats
    fancy = FancyBboxPatch((0.05, 0.05), 0.9, 0.9,
                            boxstyle="round,pad=0.02",
                            facecolor='#f8f9fa', edgecolor='#dee2e6',
                            transform=ax.transAxes, linewidth=1.5)
    ax.add_patch(fancy)

    fig.suptitle('AMRI v2 vs v1: Comprehensive Head-to-Head',
                 fontsize=15, fontweight='bold', y=1.01)
    save_fig(fig, 'FIG_v1_vs_v2_dashboard')
    plt.close(fig)


# =========================================================================
# MAIN
# =========================================================================

def main():
    print("Generating publication-quality visualizations...")
    print("=" * 60)

    figure1_convergence_proof()
    figure2_threshold_heatmap()
    figure3_blending_weights()
    figure4_generalized_coverage()
    figure5_efficiency_robustness()
    figure6_v1_vs_v2()

    print()
    print("=" * 60)
    print("All 6 figures generated successfully!")
    print(f"Output directory: {os.path.abspath(FIG_DIR)}")


if __name__ == '__main__':
    main()
