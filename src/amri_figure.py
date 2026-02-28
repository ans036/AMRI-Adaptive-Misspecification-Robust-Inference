"""Generate comprehensive AMRI comparison figure."""
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.family': 'serif',
})

FIGS = Path("c:/Users/anish/OneDrive/Desktop/Novel Research/figures")
COLORS = {
    'Naive_OLS': '#d62728', 'Sandwich_HC0': '#ff7f0e', 'Sandwich_HC3': '#2ca02c',
    'Pairs_Bootstrap': '#1f77b4', 'Wild_Bootstrap': '#9467bd', 'Bootstrap_t': '#8c564b',
    'Bayesian': '#bcbd22', 'AMRI': '#17becf',
}
LABELS = {
    'Naive_OLS': 'Naive OLS', 'Sandwich_HC0': 'HC0', 'Sandwich_HC3': 'HC3',
    'Pairs_Bootstrap': 'Pairs Boot', 'Wild_Bootstrap': 'Wild Boot',
    'Bootstrap_t': 'Boot-t', 'Bayesian': 'Bayesian', 'AMRI': 'AMRI (Proposed)',
}

# Load data
dfs = []
for f in Path("c:/Users/anish/OneDrive/Desktop/Novel Research/results").glob("*.csv"):
    dfs.append(pd.read_csv(f))
df = pd.concat(dfs, ignore_index=True)
if 'B_valid' in df.columns and 'B' not in df.columns:
    df['B'] = df['B_valid']
elif 'B_valid' in df.columns:
    df['B'] = df['B'].fillna(df['B_valid'])
df['method'] = df['method'].replace({'Bayesian_Normal': 'Bayesian'})
df = df.drop_duplicates(subset=['dgp', 'delta', 'n', 'method'], keep='last')
df = df.dropna(subset=['coverage'])

methods_order = ['Naive_OLS', 'Bayesian', 'Sandwich_HC0', 'Sandwich_HC3',
                 'Pairs_Bootstrap', 'Wild_Bootstrap', 'Bootstrap_t', 'AMRI']
methods_present = [m for m in methods_order if m in df['method'].unique()]

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# Panel A: Overall coverage bar chart
ax = axes[0, 0]
overall = df.groupby('method')['coverage'].mean().reindex(methods_present)
colors_list = [COLORS.get(m, 'gray') for m in methods_present]
ax.barh(range(len(methods_present)), overall.values, color=colors_list,
        edgecolor='black', linewidth=0.5)
ax.axvline(0.95, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Nominal 0.95')
ax.set_yticks(range(len(methods_present)))
ax.set_yticklabels([LABELS.get(m, m) for m in methods_present], fontsize=10)
ax.set_xlabel('Average Coverage Probability')
ax.set_title('A: Overall Coverage\n(higher = better)', fontweight='bold')
ax.set_xlim(0.8, 0.96)
ax.legend()
amri_idx = methods_present.index('AMRI')
ax.annotate(f'{overall["AMRI"]:.4f}', (overall["AMRI"], amri_idx),
            xytext=(15, 0), textcoords='offset points', fontweight='bold',
            fontsize=11, color='#17becf')

# Panel B: Coverage at delta=1 (robustness)
ax = axes[0, 1]
d1 = df[df['delta'] == 1.0].groupby('method')['coverage'].mean().reindex(methods_present).dropna()
ax.barh(range(len(d1)), d1.values, color=[COLORS.get(m, 'gray') for m in d1.index],
        edgecolor='black', linewidth=0.5)
ax.axvline(0.95, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.set_yticks(range(len(d1)))
ax.set_yticklabels([LABELS.get(m, m) for m in d1.index], fontsize=10)
ax.set_xlabel('Coverage at Severe Misspecification')
ax.set_title('B: Robustness (delta=1.0)\n(higher = more robust)', fontweight='bold')
ax.set_xlim(0.65, 0.97)

# Panel C: Width at delta=0 (efficiency)
ax = axes[0, 2]
d0 = df[df['delta'] == 0.0].groupby('method')['avg_width'].mean().reindex(methods_present).dropna()
ax.barh(range(len(d0)), d0.values, color=[COLORS.get(m, 'gray') for m in d0.index],
        edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(d0)))
ax.set_yticklabels([LABELS.get(m, m) for m in d0.index], fontsize=10)
ax.set_xlabel('Average Interval Width')
ax.set_title('C: Efficiency (delta=0)\n(lower = more efficient)', fontweight='bold')

# Panel D: Paired win rate
ax = axes[1, 0]
amri_df2 = df[df['method'] == 'AMRI'][['dgp', 'delta', 'n', 'coverage']].rename(
    columns={'coverage': 'cov_amri'})
win_rates = {}
for other in methods_present:
    if other == 'AMRI':
        continue
    other_df = df[df['method'] == other][['dgp', 'delta', 'n', 'coverage']].rename(
        columns={'coverage': 'cov_other'})
    m = amri_df2.merge(other_df, on=['dgp', 'delta', 'n'])
    if len(m) > 0:
        win_rates[other] = (m['cov_amri'] > m['cov_other']).mean() * 100

others = list(win_rates.keys())
vals = [win_rates[m] for m in others]
colors_wr = [COLORS.get(m, 'gray') for m in others]
ax.barh(range(len(others)), vals, color=colors_wr, edgecolor='black', linewidth=0.5)
ax.axvline(50, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.set_yticks(range(len(others)))
ax.set_yticklabels([LABELS.get(m, m) for m in others], fontsize=10)
ax.set_xlabel('Win Rate (%)')
ax.set_title('D: AMRI Win Rate vs Each Method\n(% scenarios with higher coverage)', fontweight='bold')
ax.set_xlim(0, 105)
for i, v in enumerate(vals):
    ax.text(v + 1, i, f'{v:.0f}%', va='center', fontweight='bold', fontsize=10)

# Panel E: Coverage by delta (the key plot)
ax = axes[1, 1]
for method in ['Naive_OLS', 'Sandwich_HC3', 'AMRI']:
    msub = df[df['method'] == method]
    avg = msub.groupby('delta')['coverage'].mean().reset_index().sort_values('delta')
    style = '--' if method == 'Naive_OLS' else '-'
    lw = 3.5 if method == 'AMRI' else 2
    ax.plot(avg['delta'], avg['coverage'], f'o{style}',
            color=COLORS[method], label=LABELS[method],
            linewidth=lw, markersize=8)
ax.axhline(0.95, color='black', linestyle=':', linewidth=1)
ax.axhspan(0.935, 0.965, alpha=0.06, color='green')
ax.fill_between([0, 1], 0, 0.93, alpha=0.03, color='red')
ax.set_xlabel('Misspecification Severity (delta)')
ax.set_ylabel('Coverage')
ax.set_title('E: Coverage vs Severity\n(AMRI stays near 0.95 throughout)', fontweight='bold')
ax.set_ylim(0.7, 0.97)
ax.legend(fontsize=11, loc='lower left')
ax.grid(True, alpha=0.15)

# Panel F: Coverage by sample size
ax = axes[1, 2]
for method in ['Naive_OLS', 'Sandwich_HC3', 'AMRI']:
    msub = df[(df['method'] == method) & (df['delta'] >= 0.5)]
    avg = msub.groupby('n')['coverage'].mean().reset_index().sort_values('n')
    style = '--' if method == 'Naive_OLS' else '-'
    lw = 3.5 if method == 'AMRI' else 2
    ax.plot(avg['n'], avg['coverage'], f'o{style}',
            color=COLORS[method], label=LABELS[method],
            linewidth=lw, markersize=8)
ax.axhline(0.95, color='black', linestyle=':', linewidth=1)
ax.set_xlabel('Sample Size (n)')
ax.set_ylabel('Coverage (at delta >= 0.5)')
ax.set_title('F: Sample Size Paradox\n(more data hurts Naive, helps AMRI)', fontweight='bold')
ax.set_xscale('log')
ax.set_ylim(0.7, 0.97)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.15)

fig.suptitle('AMRI: Adaptive Misspecification-Robust Inference\n'
             'Comprehensive Performance Analysis (240 scenarios, 2000 reps each)',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(FIGS / 'AMRI_comprehensive.png', bbox_inches='tight')
fig.savefig(FIGS / 'AMRI_comprehensive.pdf', bbox_inches='tight')
plt.close()
print("Saved: AMRI_comprehensive.png + .pdf")
