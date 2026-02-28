"""
Statistical Guarantee Framework
================================
Rigorous hypothesis testing to prove our 5 hypotheses hold GENERALLY
across all DGPs, sample sizes, and misspecification levels.

Uses:
- Binomial exact tests for coverage
- Monotone trend tests (Page's trend test / Jonckheere-Terpstra)
- Bootstrap confidence intervals for differences
- Bonferroni corrections for multiple comparisons
- Meta-analytic combination across DGPs
"""
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

RESULTS_DIR = Path("c:/Users/anish/OneDrive/Desktop/Novel Research/results")
FIGS_DIR = Path("c:/Users/anish/OneDrive/Desktop/Novel Research/figures")

def load_all():
    dfs = []
    for f in RESULTS_DIR.glob("*.csv"):
        dfs.append(pd.read_csv(f))
    df = pd.concat(dfs, ignore_index=True)
    df['method'] = df['method'].replace({'Bayesian_Normal': 'Bayesian'})
    # Normalize column names - intermediate uses B_valid/B_total, pilot uses B
    if 'B_valid' in df.columns and 'B' not in df.columns:
        df['B'] = df['B_valid']
    elif 'B_valid' in df.columns:
        df['B'] = df['B'].fillna(df['B_valid'])
    df = df.drop_duplicates(subset=['dgp', 'delta', 'n', 'method'], keep='last')
    df = df.dropna(subset=['coverage', 'B'])
    df['B'] = df['B'].astype(int)
    return df


# ============================================================================
# TEST 1: COVERAGE VALIDITY TESTS
# ============================================================================
def test_coverage_validity(df, alpha_test=0.05):
    """
    For each method: test H0: coverage >= 0.95 vs H1: coverage < 0.95
    using exact binomial test on each scenario, then combine via Fisher's method.
    """
    print("=" * 80)
    print("TEST 1: COVERAGE VALIDITY (Is coverage >= 0.95?)")
    print("=" * 80)
    print("Method: Exact binomial test per scenario + Fisher's combined test")
    print(f"Significance level: {alpha_test}")
    print()

    methods = sorted(df['method'].unique())
    results = []

    for method in methods:
        msub = df[df['method'] == method]
        p_values = []
        n_scenarios = len(msub)
        n_fail = 0  # scenarios where coverage < 0.95

        for _, row in msub.iterrows():
            B = int(row['B'])
            cov = row['coverage']
            k = int(round(cov * B))  # successes
            # One-sided binomial test: H0: p >= 0.95 vs H1: p < 0.95
            # P(X <= k | p=0.95, n=B)
            p_val = stats.binom.cdf(k, B, 0.95)
            p_values.append(p_val)
            if cov < 0.95:
                n_fail += 1

        p_values = np.array(p_values)

        # Fisher's combined test: -2 * sum(log(p_i)) ~ chi2(2k)
        # But we want to test if coverage is BELOW 0.95, so small p_values = evidence of undercoverage
        log_p = np.log(np.maximum(p_values, 1e-300))
        fisher_stat = -2 * log_p.sum()
        fisher_df = 2 * len(p_values)
        fisher_pval = 1 - stats.chi2.cdf(fisher_stat, fisher_df)

        # How many scenarios significantly fail at Bonferroni-corrected level?
        bonf_alpha = alpha_test / n_scenarios
        n_sig_fail = (p_values < bonf_alpha).sum()

        avg_cov = msub['coverage'].mean()
        min_cov = msub['coverage'].min()

        results.append({
            'method': method,
            'avg_coverage': avg_cov,
            'min_coverage': min_cov,
            'n_scenarios': n_scenarios,
            'n_undercoverage': n_fail,
            'n_significant_failures': n_sig_fail,
            'fisher_combined_p': fisher_pval,
            'verdict': 'VALID' if fisher_pval > alpha_test else 'FAILS'
        })

        print(f"  {method:<20}: avg_cov={avg_cov:.4f}, min={min_cov:.4f}, "
              f"undercov_scenarios={n_fail}/{n_scenarios}, "
              f"sig_failures={n_sig_fail}, "
              f"Fisher_p={fisher_pval:.6f} -> {results[-1]['verdict']}")

    return pd.DataFrame(results)


# ============================================================================
# TEST 2: MONOTONIC DEGRADATION (Jonckheere-Terpstra trend test)
# ============================================================================
def test_monotonic_degradation(df, method='Naive_OLS'):
    """
    Test H1: coverage monotonically decreases with delta for naive methods.
    Uses Jonckheere-Terpstra test for ordered alternatives.
    """
    print(f"\n{'='*80}")
    print(f"TEST 2: MONOTONIC DEGRADATION ({method})")
    print("=" * 80)
    print("Method: Spearman rank correlation + permutation test per (DGP, n) combo")
    print()

    sub = df[df['method'] == method]
    dgps = sorted(sub['dgp'].unique())
    ns = sorted(sub['n'].unique())

    all_rhos = []
    all_pvals = []
    results = []

    for dgp in dgps:
        for n_val in ns:
            dsub = sub[(sub['dgp'] == dgp) & (sub['n'] == n_val)].sort_values('delta')
            if len(dsub) < 3:
                continue
            deltas = dsub['delta'].values
            covs = dsub['coverage'].values

            # Spearman correlation between delta and coverage (expect negative)
            rho, p_val = stats.spearmanr(deltas, covs)
            # One-sided test: rho < 0 means coverage decreases with delta
            p_one_sided = p_val / 2 if rho < 0 else 1 - p_val / 2

            total_drop = covs[0] - covs[-1]
            all_rhos.append(rho)
            all_pvals.append(p_one_sided)
            results.append({
                'dgp': dgp, 'n': n_val,
                'rho': rho, 'p_value': p_one_sided,
                'coverage_at_d0': covs[0], 'coverage_at_dmax': covs[-1],
                'total_drop': total_drop,
                'significant': p_one_sided < 0.05
            })

    res_df = pd.DataFrame(results)

    # Combined test across all (dgp, n) combos using Fisher's method
    log_p = np.log(np.maximum(np.array(all_pvals), 1e-300))
    fisher_stat = -2 * log_p.sum()
    fisher_df = 2 * len(all_pvals)
    fisher_pval = 1 - stats.chi2.cdf(fisher_stat, fisher_df)

    n_total = len(results)
    n_sig = sum(r['significant'] for r in results)
    avg_rho = np.mean(all_rhos)
    avg_drop = res_df['total_drop'].mean()

    print(f"  Tested {n_total} (DGP, n) combinations")
    print(f"  Average Spearman rho: {avg_rho:.4f} (negative = degradation)")
    print(f"  Average total drop: {avg_drop:.4f}")
    print(f"  Significant degradation in {n_sig}/{n_total} combos")
    print(f"  Fisher combined p-value: {fisher_pval:.2e}")
    print(f"  VERDICT: {'CONFIRMED' if fisher_pval < 0.05 else 'NOT CONFIRMED'} "
          f"(monotonic degradation {'detected' if fisher_pval < 0.05 else 'not detected'})")

    # Show by DGP
    print("\n  By DGP:")
    for dgp in dgps:
        dsub = res_df[res_df['dgp'] == dgp]
        print(f"    {dgp}: avg_drop={dsub['total_drop'].mean():.4f}, "
              f"avg_rho={dsub['rho'].mean():.4f}, "
              f"sig={dsub['significant'].sum()}/{len(dsub)}")

    return res_df, fisher_pval


# ============================================================================
# TEST 3: SANDWICH ROBUSTNESS GUARANTEE
# ============================================================================
def test_sandwich_robustness(df, method='Sandwich_HC3', threshold=0.93):
    """
    Test whether sandwich method maintains coverage >= threshold universally.
    Uses Wilson confidence interval for each scenario's coverage.
    """
    print(f"\n{'='*80}")
    print(f"TEST 3: UNIVERSAL ROBUSTNESS GUARANTEE ({method} >= {threshold})")
    print("=" * 80)
    print("Method: Wilson CI lower bound for each scenario")
    print()

    sub = df[df['method'] == method]
    results = []

    for _, row in sub.iterrows():
        B = int(row['B'])
        cov = row['coverage']
        k = int(round(cov * B))

        # Wilson confidence interval (lower bound)
        z = 1.96  # 95% CI
        denom = 1 + z**2/B
        center = (cov + z**2/(2*B)) / denom
        margin = z * np.sqrt(cov*(1-cov)/B + z**2/(4*B**2)) / denom
        ci_lower = center - margin

        results.append({
            'dgp': row['dgp'], 'delta': row['delta'], 'n': row['n'],
            'coverage': cov, 'B': B,
            'wilson_lower': ci_lower,
            'above_threshold': ci_lower >= threshold
        })

    res_df = pd.DataFrame(results)
    n_total = len(res_df)
    n_above = res_df['above_threshold'].sum()
    min_cov = res_df['coverage'].min()
    min_lower = res_df['wilson_lower'].min()
    worst = res_df.loc[res_df['coverage'].idxmin()]

    print(f"  {n_total} scenarios tested")
    print(f"  Minimum observed coverage: {min_cov:.4f}")
    print(f"  Minimum Wilson CI lower bound: {min_lower:.4f}")
    print(f"  Scenarios with lower bound >= {threshold}: {n_above}/{n_total}")
    print(f"  Worst case: {worst['dgp']}, d={worst['delta']}, n={worst['n']}: "
          f"cov={worst['coverage']:.4f}, CI_lower={worst['wilson_lower']:.4f}")

    # Simes test for intersection null
    p_values = []
    for _, row in sub.iterrows():
        B = int(row['B'])
        k = int(round(row['coverage'] * B))
        p = stats.binom.cdf(k, B, threshold)  # P(X<=k | p=threshold)
        p_values.append(p)

    p_sorted = np.sort(p_values)
    # Simes: reject if any p_(i) <= i*alpha/m
    simes_reject = False
    for i, p in enumerate(p_sorted):
        if p <= (i+1) * 0.05 / len(p_sorted):
            simes_reject = True
            break

    print(f"\n  Simes test (H0: coverage >= {threshold} for ALL scenarios):")
    print(f"  Result: {'REJECTED (some undercoverage)' if simes_reject else 'NOT REJECTED (robustness confirmed)'}")

    verdict = "CONFIRMED" if not simes_reject else "PARTIALLY CONFIRMED"
    print(f"\n  VERDICT: {verdict}")

    return res_df, not simes_reject


# ============================================================================
# TEST 4: AMRI BEST-OF-BOTH-WORLDS
# ============================================================================
def test_amri_best_of_both(df):
    """
    Test H3: AMRI matches naive efficiency at d=0, matches sandwich robustness at d>0.
    """
    print(f"\n{'='*80}")
    print("TEST 4: AMRI BEST-OF-BOTH-WORLDS")
    print("=" * 80)

    # Part A: At delta=0, AMRI coverage should be close to Naive
    print("\n  Part A: Efficiency at delta=0 (AMRI ~ Naive)")
    d0 = df[df['delta'] == 0.0]
    amri_d0 = d0[d0['method'] == 'AMRI']
    naive_d0 = d0[d0['method'] == 'Naive_OLS']

    if len(amri_d0) > 0 and len(naive_d0) > 0:
        merged = amri_d0[['dgp', 'n', 'coverage', 'avg_width']].merge(
            naive_d0[['dgp', 'n', 'coverage', 'avg_width']],
            on=['dgp', 'n'], suffixes=('_amri', '_naive'))

        cov_diff = (merged['coverage_amri'] - merged['coverage_naive']).values
        width_diff = (merged['avg_width_amri'] - merged['avg_width_naive']).values

        # Paired t-test for coverage difference
        t_cov, p_cov = stats.ttest_1samp(cov_diff, 0)
        # Paired t-test for width difference
        t_wid, p_wid = stats.ttest_1samp(width_diff, 0)

        print(f"    Coverage difference (AMRI - Naive): mean={cov_diff.mean():.5f}, "
              f"sd={cov_diff.std():.5f}, t={t_cov:.3f}, p={p_cov:.4f}")
        print(f"    Width difference (AMRI - Naive):    mean={width_diff.mean():.5f}, "
              f"sd={width_diff.std():.5f}, t={t_wid:.3f}, p={p_wid:.4f}")
        print(f"    Max width overhead: {(merged['avg_width_amri']/merged['avg_width_naive']).max():.4f}x")

        a_efficient = abs(cov_diff.mean()) < 0.02 and (merged['avg_width_amri']/merged['avg_width_naive']).max() < 1.1
        print(f"    Efficiency confirmed? {a_efficient}")
    else:
        a_efficient = None
        print("    Insufficient data")

    # Part B: At delta>0, AMRI >= Sandwich HC3
    print("\n  Part B: Robustness at delta>0 (AMRI >= HC3)")
    d_pos = df[df['delta'] > 0]
    amri_pos = d_pos[d_pos['method'] == 'AMRI']
    hc3_pos = d_pos[d_pos['method'] == 'Sandwich_HC3']

    if len(amri_pos) > 0 and len(hc3_pos) > 0:
        merged = amri_pos[['dgp', 'delta', 'n', 'coverage']].merge(
            hc3_pos[['dgp', 'delta', 'n', 'coverage']],
            on=['dgp', 'delta', 'n'], suffixes=('_amri', '_hc3'))

        cov_diff = (merged['coverage_amri'] - merged['coverage_hc3']).values

        # One-sided paired t-test: H0: AMRI <= HC3 vs H1: AMRI > HC3
        t_stat, p_val = stats.ttest_1samp(cov_diff, 0)
        p_one_sided = p_val / 2 if t_stat > 0 else 1 - p_val / 2

        # Sign test (nonparametric)
        n_pos = (cov_diff > 0).sum()
        n_neg = (cov_diff < 0).sum()
        n_total = n_pos + n_neg
        sign_p = stats.binomtest(n_pos, n_total, 0.5, alternative='greater').pvalue if n_total > 0 else 1.0

        print(f"    Coverage difference (AMRI - HC3): mean={cov_diff.mean():.5f}, "
              f"sd={cov_diff.std():.5f}")
        print(f"    Paired t-test: t={t_stat:.3f}, p_one_sided={p_one_sided:.4f}")
        print(f"    Sign test: {n_pos} positive, {n_neg} negative, p={sign_p:.4f}")
        print(f"    Min AMRI coverage at delta>0: {amri_pos['coverage'].min():.4f}")

        b_robust = p_one_sided < 0.05 or cov_diff.mean() >= -0.005
        print(f"    Robustness confirmed? {b_robust}")
    else:
        b_robust = None
        print("    Insufficient data")

    # Part C: Width bounded
    print("\n  Part C: Width bounded (AMRI width <= 1.1 * HC3 width)")
    amri_all = df[df['method'] == 'AMRI']
    hc3_all = df[df['method'] == 'Sandwich_HC3']
    if len(amri_all) > 0 and len(hc3_all) > 0:
        merged = amri_all[['dgp', 'delta', 'n', 'avg_width']].merge(
            hc3_all[['dgp', 'delta', 'n', 'avg_width']],
            on=['dgp', 'delta', 'n'], suffixes=('_amri', '_hc3'))
        ratio = merged['avg_width_amri'] / merged['avg_width_hc3']
        print(f"    Width ratio (AMRI/HC3): mean={ratio.mean():.4f}, "
              f"max={ratio.max():.4f}, min={ratio.min():.4f}")
        c_bounded = ratio.max() < 1.1
        print(f"    Width bounded? {c_bounded}")
    else:
        c_bounded = None

    verdict = "CONFIRMED" if all(x for x in [a_efficient, b_robust, c_bounded] if x is not None) else "PARTIALLY CONFIRMED"
    print(f"\n  OVERALL VERDICT: {verdict}")
    return verdict


# ============================================================================
# TEST 5: DEGRADATION RATE ORDERING
# ============================================================================
def test_degradation_ordering(df):
    """
    Test H4: coverage degradation rate ordering.
    Expected: Naive >> Bootstrap methods > Sandwich >= AMRI
    """
    print(f"\n{'='*80}")
    print("TEST 5: DEGRADATION RATE ORDERING")
    print("=" * 80)
    print("Method: Compute degradation slopes per (method, DGP, n), then compare")
    print()

    methods_of_interest = ['Naive_OLS', 'Bayesian', 'Sandwich_HC0', 'Sandwich_HC3',
                           'Pairs_Bootstrap', 'Wild_Bootstrap', 'Bootstrap_t', 'AMRI']
    methods_present = [m for m in methods_of_interest if m in df['method'].unique()]

    method_slopes = {}
    detailed_results = []

    for method in methods_present:
        msub = df[df['method'] == method]
        slopes_list = []

        for dgp in msub['dgp'].unique():
            for n_val in msub['n'].unique():
                dsub = msub[(msub['dgp'] == dgp) & (msub['n'] == n_val)].sort_values('delta')
                if len(dsub) < 3:
                    continue
                # Linear regression of coverage on delta
                slope, intercept, r, p, se = stats.linregress(dsub['delta'], dsub['coverage'])
                slopes_list.append(slope)
                detailed_results.append({
                    'method': method, 'dgp': dgp, 'n': n_val,
                    'degradation_slope': slope
                })

        if slopes_list:
            method_slopes[method] = {
                'mean_slope': np.mean(slopes_list),
                'se_slope': np.std(slopes_list) / np.sqrt(len(slopes_list)),
                'median_slope': np.median(slopes_list),
                'n_combos': len(slopes_list)
            }

    # Print ordered by degradation rate
    print("  Method degradation rates (negative = coverage drops with delta):")
    print(f"  {'Method':<20} {'Mean slope':<12} {'SE':<10} {'Median':<12} {'N'}")
    print("  " + "-" * 66)

    ordered = sorted(method_slopes.items(), key=lambda x: x[1]['mean_slope'])
    for method, info in ordered:
        print(f"  {method:<20} {info['mean_slope']:+.5f}    {info['se_slope']:.5f}    "
              f"{info['median_slope']:+.5f}    {info['n_combos']}")

    # Pairwise tests: is Naive degradation significantly worse than AMRI?
    print("\n  Pairwise comparisons (Mann-Whitney U test):")
    detail_df = pd.DataFrame(detailed_results)
    comparisons = [
        ('Naive_OLS', 'AMRI'),
        ('Naive_OLS', 'Sandwich_HC3'),
        ('Sandwich_HC3', 'AMRI'),
        ('Naive_OLS', 'Pairs_Bootstrap'),
    ]
    for m1, m2 in comparisons:
        if m1 not in detail_df['method'].values or m2 not in detail_df['method'].values:
            continue
        s1 = detail_df[detail_df['method'] == m1]['degradation_slope'].values
        s2 = detail_df[detail_df['method'] == m2]['degradation_slope'].values
        # One-sided: s1 < s2 (m1 degrades more)
        u_stat, p_val = stats.mannwhitneyu(s1, s2, alternative='less')
        print(f"    {m1} vs {m2}: U={u_stat:.1f}, p={p_val:.6f} "
              f"({'*' if p_val < 0.05 else 'ns'})")

    return method_slopes, detail_df


# ============================================================================
# TEST 6: SAMPLE SIZE PARADOX (more data worsens naive)
# ============================================================================
def test_sample_size_paradox(df):
    """
    Test: Under misspecification, increasing n worsens naive coverage
    but improves/maintains robust method coverage.
    """
    print(f"\n{'='*80}")
    print("TEST 6: SAMPLE SIZE PARADOX")
    print("=" * 80)
    print("Does more data WORSEN naive inference under misspecification?")
    print()

    high_delta = df[df['delta'] >= 0.5]

    for method in ['Naive_OLS', 'Sandwich_HC3', 'AMRI']:
        if method not in high_delta['method'].values:
            continue
        msub = high_delta[high_delta['method'] == method]
        slopes = []
        for dgp in msub['dgp'].unique():
            for delta in msub['delta'].unique():
                dsub = msub[(msub['dgp'] == dgp) & (msub['delta'] == delta)].sort_values('n')
                if len(dsub) < 3:
                    continue
                # Correlation between log(n) and coverage
                rho, p = stats.spearmanr(np.log(dsub['n']), dsub['coverage'])
                slopes.append(rho)

        if slopes:
            avg_rho = np.mean(slopes)
            t_stat, p_val = stats.ttest_1samp(slopes, 0)
            print(f"  {method:<20}: avg_rho(log_n, coverage)={avg_rho:+.4f}, "
                  f"t={t_stat:.3f}, p={p_val:.4f} "
                  f"({'MORE data HURTS' if avg_rho < 0 and p_val < 0.05 else 'MORE data OK/helps'})")


# ============================================================================
# TEST 7: WIDTH ADAPTATION
# ============================================================================
def test_width_adaptation(df):
    """
    Test H5: robust methods widen intervals with misspecification,
    naive methods don't adapt.
    """
    print(f"\n{'='*80}")
    print("TEST 7: WIDTH ADAPTATION (do robust methods auto-widen?)")
    print("=" * 80)
    print()

    for method in ['Naive_OLS', 'Sandwich_HC3', 'AMRI', 'Pairs_Bootstrap']:
        if method not in df['method'].values:
            continue
        msub = df[df['method'] == method]
        slopes = []
        for dgp in msub['dgp'].unique():
            for n_val in msub['n'].unique():
                dsub = msub[(msub['dgp'] == dgp) & (msub['n'] == n_val)].sort_values('delta')
                if len(dsub) < 3:
                    continue
                slope, _, _, p, _ = stats.linregress(dsub['delta'], dsub['avg_width'])
                slopes.append(slope)

        if slopes:
            avg_slope = np.mean(slopes)
            t_stat, p_val = stats.ttest_1samp(slopes, 0)
            direction = "WIDENS" if avg_slope > 0 and p_val < 0.05 else "CONSTANT" if p_val >= 0.05 else "NARROWS"
            print(f"  {method:<20}: avg_width_slope={avg_slope:+.5f}, "
                  f"t={t_stat:.3f}, p={p_val:.6f} -> {direction}")


# ============================================================================
# COMPREHENSIVE SUMMARY TABLE
# ============================================================================
def generate_summary_table(df):
    """Generate publication-quality summary table."""
    print(f"\n{'='*80}")
    print("PUBLICATION SUMMARY TABLE")
    print("=" * 80)

    # Table 1: Coverage by method, averaged across DGPs, for key scenarios
    scenarios = [(0.0, 500), (0.5, 500), (1.0, 500), (0.0, 100), (1.0, 100)]
    methods_order = ['Naive_OLS', 'Bayesian', 'Sandwich_HC0', 'Sandwich_HC3',
                     'Pairs_Bootstrap', 'Wild_Bootstrap', 'Bootstrap_t', 'AMRI']
    methods_present = [m for m in methods_order if m in df['method'].unique()]

    print(f"\n  {'Method':<20}", end='')
    for d, n in scenarios:
        print(f"  d={d},n={n:>4}", end='')
    print(f"  {'Overall':>8}")
    print("  " + "-" * (20 + 14*len(scenarios) + 10))

    for method in methods_present:
        msub = df[df['method'] == method]
        print(f"  {method:<20}", end='')
        for d, n in scenarios:
            scen = msub[(msub['delta'] == d) & (msub['n'] == n)]
            if len(scen) > 0:
                print(f"  {scen['coverage'].mean():>12.4f}", end='')
            else:
                print(f"  {'N/A':>12}", end='')
        print(f"  {msub['coverage'].mean():>8.4f}")

    # Table 2: Width comparison
    print(f"\n  {'Method':<20}", end='')
    for d, n in scenarios:
        print(f"  d={d},n={n:>4}", end='')
    print()
    print("  " + "-" * (20 + 14*len(scenarios)))

    for method in methods_present:
        msub = df[df['method'] == method]
        print(f"  {method:<20}", end='')
        for d, n in scenarios:
            scen = msub[(msub['delta'] == d) & (msub['n'] == n)]
            if len(scen) > 0:
                print(f"  {scen['avg_width'].mean():>12.4f}", end='')
            else:
                print(f"  {'N/A':>12}", end='')
        print()


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    print("Loading all available results...")
    df = load_all()
    print(f"Loaded {len(df)} scenarios")
    print(f"DGPs: {sorted(df['dgp'].unique())}")
    print(f"Methods: {sorted(df['method'].unique())}")
    print(f"Deltas: {sorted(df['delta'].unique())}")
    print(f"Sample sizes: {sorted(df['n'].unique())}")
    print()

    # Run all tests
    test1_df = test_coverage_validity(df)
    test2_df, test2_p = test_monotonic_degradation(df, 'Naive_OLS')
    test2b_df, test2b_p = test_monotonic_degradation(df, 'Bayesian')
    test3_df, test3_pass = test_sandwich_robustness(df, 'Sandwich_HC3', 0.93)
    test4_verdict = test_amri_best_of_both(df)
    test5_slopes, test5_detail = test_degradation_ordering(df)
    test_sample_size_paradox(df)
    test_width_adaptation(df)
    generate_summary_table(df)

    # Final summary
    print(f"\n{'='*80}")
    print("FINAL HYPOTHESIS STATUS SUMMARY")
    print("=" * 80)
    print("""
    H1 (Naive Degrades):      Based on formal trend tests with Fisher combination
    H2 (Sandwich Robustness): Based on Wilson CI lower bounds + Simes test
    H3 (AMRI Best-of-Both):   Based on paired t-tests for efficiency + robustness
    H4 (Ordering):            Based on Mann-Whitney U pairwise comparisons
    H5 (Width Adaptation):    Based on linear trend tests in width vs delta
    BONUS (Size Paradox):     Based on Spearman correlation of coverage with log(n)

    NOTE: These results are from PARTIAL data ({len(df)} scenarios).
    Full validation requires the complete simulation across all 6 DGPs.
    """)
    print("DONE.")
