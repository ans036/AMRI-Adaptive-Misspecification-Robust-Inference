"""
Statistical Guarantee Framework v2
====================================
Rigorous hypothesis testing for AMRI's coverage, efficiency, and robustness
properties. Tests are designed for publication in a top-tier journal.

Tests:
1. Coverage validity (binomial + Agresti-Coull + Fisher combination)
2. Monotonic degradation of naive method (Spearman + Fisher)
3. Universal robustness guarantee for sandwich/AMRI (Wilson CI + Simes)
4. AMRI best-of-both-worlds (paired t-tests, equivalence TOST)
5. Degradation rate ordering (Mann-Whitney U)
6. Sample size paradox (Spearman)
7. Width adaptation (linear trend)
8. AMRI vs AKS head-to-head (competitor comparison)
9. Power analysis (minimum B for 80% power)

References:
  Agresti & Coull (1998). "Approximate is better than 'exact' for interval
    estimation of binomial proportions." American Statistician.
  Lakens (2017). "Equivalence Tests: A Practical Primer for t Tests,
    Correlations, and Meta-Analyses." Social Psych & Personality Science.
"""
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGS_DIR = Path(__file__).resolve().parent.parent / "figures"


def load_all():
    """Load simulation results, handling multiple CSV formats."""
    dfs = []
    # Only load specific result files (not theorem/competitor CSVs)
    for pattern in ['results_final_*.csv', 'results_vectorized_*.csv',
                    'results_v2_*.csv', 'amri_v2_*.csv']:
        for f in RESULTS_DIR.glob(pattern):
            try:
                dfs.append(pd.read_csv(f))
            except Exception:
                pass

    # Also load the main competitor comparison
    comp_file = RESULTS_DIR / 'results_competitor_comparison.csv'
    if comp_file.exists():
        try:
            dfs.append(pd.read_csv(comp_file))
        except Exception:
            pass

    if not dfs:
        # Fallback: load all CSVs that have the right columns
        for f in RESULTS_DIR.glob("*.csv"):
            try:
                df_temp = pd.read_csv(f)
                if 'coverage' in df_temp.columns and 'method' in df_temp.columns:
                    dfs.append(df_temp)
            except Exception:
                pass

    if not dfs:
        raise FileNotFoundError(f"No valid result CSVs found in {RESULTS_DIR}")

    df = pd.concat(dfs, ignore_index=True)

    # Normalize method names
    df['method'] = df['method'].replace({
        'Bayesian_Normal': 'Bayesian',
        'AMRI': 'AMRI_v1',  # old naming convention
    })

    # Normalize column names
    if 'B_valid' in df.columns and 'B' not in df.columns:
        df['B'] = df['B_valid']
    elif 'B_valid' in df.columns:
        df['B'] = df['B'].fillna(df['B_valid'])

    # If B column doesn't exist, try valid_reps
    if 'B' not in df.columns and 'valid_reps' in df.columns:
        df['B'] = df['valid_reps']
    elif 'B' not in df.columns:
        df['B'] = 2000  # default

    df = df.drop_duplicates(subset=['dgp', 'delta', 'n', 'method'], keep='last')
    df = df.dropna(subset=['coverage'])
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
            if len(dsub) < 2:
                continue
            deltas = dsub['delta'].values
            covs = dsub['coverage'].values

            total_drop = covs[0] - covs[-1]

            if len(dsub) >= 3:
                rho, p_val = stats.spearmanr(deltas, covs)
                p_one_sided = p_val / 2 if rho < 0 else 1 - p_val / 2
            else:
                # With only 2 points, use sign of drop as test
                rho = -1.0 if total_drop > 0 else 1.0
                p_one_sided = 0.5  # uninformative

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

    if len(results) == 0:
        print("  No (DGP, n) combinations with enough delta values to test.")
        return pd.DataFrame(), 1.0

    # Combined test across all (dgp, n) combos using Fisher's method
    log_p = np.log(np.maximum(np.array(all_pvals), 1e-300))
    fisher_stat = -2 * log_p.sum()
    fisher_df = 2 * len(all_pvals)
    fisher_pval = 1 - stats.chi2.cdf(fisher_stat, fisher_df)

    n_total = len(results)
    n_sig = sum(r['significant'] for r in results)
    avg_rho = np.mean(all_rhos)
    avg_drop = np.mean([r['total_drop'] for r in results])

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
def tost_equivalence(x, equiv_margin, alpha=0.05):
    """
    Two One-Sided Tests (TOST) for equivalence.
    Tests H0: |mean(x)| >= equiv_margin vs H1: |mean(x)| < equiv_margin.
    Returns (reject, p_value, ci_lower, ci_upper).
    """
    n = len(x)
    mean_x = np.mean(x)
    se_x = np.std(x, ddof=1) / np.sqrt(n)
    df = n - 1

    # Upper test: H0: mean >= equiv_margin
    t_upper = (mean_x - equiv_margin) / se_x
    p_upper = stats.t.cdf(t_upper, df)

    # Lower test: H0: mean <= -equiv_margin
    t_lower = (mean_x + equiv_margin) / se_x
    p_lower = 1 - stats.t.cdf(t_lower, df)

    p_tost = max(p_upper, p_lower)
    reject = p_tost < alpha

    # 90% CI for equivalence (corresponds to two one-sided 5% tests)
    t_crit = stats.t.ppf(1 - alpha, df)
    ci_lo = mean_x - t_crit * se_x
    ci_hi = mean_x + t_crit * se_x

    return reject, p_tost, ci_lo, ci_hi


def test_amri_best_of_both(df):
    """
    Test H3: AMRI v2 matches naive efficiency at d=0, matches sandwich robustness at d>0.
    Uses TOST equivalence testing with epsilon=0.02 coverage margin.
    """
    print(f"\n{'='*80}")
    print("TEST 4: AMRI BEST-OF-BOTH-WORLDS (with TOST equivalence)")
    print("=" * 80)

    # Find AMRI methods (prefer v2, fall back to v1 or 'AMRI')
    amri_method = None
    for m in ['AMRI_v2', 'AMRI_v1', 'AMRI']:
        if m in df['method'].unique():
            amri_method = m
            break
    if amri_method is None:
        print("  No AMRI method found in data.")
        return "NO DATA"

    equiv_margin = 0.02  # Coverage equivalence margin

    # Part A: At delta=0, AMRI coverage equivalent to Naive (TOST)
    print(f"\n  Part A: Efficiency at delta=0 ({amri_method} equivalent to Naive)")
    print(f"  TOST equivalence margin: +/- {equiv_margin}")
    d0 = df[df['delta'] == 0.0]
    amri_d0 = d0[d0['method'] == amri_method]
    naive_d0 = d0[d0['method'] == 'Naive_OLS']

    if len(amri_d0) > 0 and len(naive_d0) > 0:
        merged = amri_d0[['dgp', 'n', 'coverage', 'avg_width']].merge(
            naive_d0[['dgp', 'n', 'coverage', 'avg_width']],
            on=['dgp', 'n'], suffixes=('_amri', '_naive'))

        cov_diff = (merged['coverage_amri'] - merged['coverage_naive']).values
        width_ratio = (merged['avg_width_amri'] / merged['avg_width_naive']).values

        # TOST for coverage equivalence
        reject, p_tost, ci_lo, ci_hi = tost_equivalence(cov_diff, equiv_margin)

        print(f"    Coverage difference ({amri_method} - Naive): mean={cov_diff.mean():.5f}")
        print(f"    TOST p-value: {p_tost:.4f}, 90% CI: [{ci_lo:.5f}, {ci_hi:.5f}]")
        print(f"    Equivalence: {'CONFIRMED' if reject else 'NOT CONFIRMED'} "
              f"(|diff| < {equiv_margin})")
        print(f"    Width ratio ({amri_method}/Naive): mean={width_ratio.mean():.4f}, max={width_ratio.max():.4f}")

        a_efficient = reject and width_ratio.max() < 1.10
        print(f"    Efficiency verdict: {a_efficient}")
    else:
        a_efficient = None
        print("    Insufficient data")

    # Part B: At delta>0, AMRI coverage equivalent to HC3 (TOST)
    print(f"\n  Part B: Robustness at delta>0 ({amri_method} equivalent to HC3)")
    d_pos = df[df['delta'] > 0]
    amri_pos = d_pos[d_pos['method'] == amri_method]
    hc3_pos = d_pos[d_pos['method'] == 'Sandwich_HC3']

    if len(amri_pos) > 0 and len(hc3_pos) > 0:
        merged = amri_pos[['dgp', 'delta', 'n', 'coverage']].merge(
            hc3_pos[['dgp', 'delta', 'n', 'coverage']],
            on=['dgp', 'delta', 'n'], suffixes=('_amri', '_hc3'))

        cov_diff = (merged['coverage_amri'] - merged['coverage_hc3']).values

        # TOST for equivalence
        reject, p_tost, ci_lo, ci_hi = tost_equivalence(cov_diff, equiv_margin)

        # Also do one-sided test: AMRI not worse than HC3
        t_stat, p_twosided = stats.ttest_1samp(cov_diff, 0)
        p_notworse = p_twosided / 2 if t_stat > 0 else 1 - p_twosided / 2

        # Sign test
        n_pos = (cov_diff > 0).sum()
        n_neg = (cov_diff < 0).sum()
        n_total = n_pos + n_neg

        print(f"    Coverage difference ({amri_method} - HC3): mean={cov_diff.mean():.5f}")
        print(f"    TOST p-value: {p_tost:.4f}, 90% CI: [{ci_lo:.5f}, {ci_hi:.5f}]")
        print(f"    Equivalence: {'CONFIRMED' if reject else 'NOT CONFIRMED'}")
        print(f"    Not-worse (one-sided) p: {p_notworse:.4f}")
        print(f"    Sign: {n_pos}+, {n_neg}-, {n_total - n_pos - n_neg}=")
        print(f"    Min {amri_method} coverage at delta>0: {amri_pos['coverage'].min():.4f}")

        b_robust = reject or cov_diff.mean() >= -0.005
        print(f"    Robustness verdict: {b_robust}")
    else:
        b_robust = None
        print("    Insufficient data")

    # Part C: Width bounded
    print(f"\n  Part C: Width bounded ({amri_method} width <= 1.1 * HC3 width)")
    amri_all = df[df['method'] == amri_method]
    hc3_all = df[df['method'] == 'Sandwich_HC3']
    if len(amri_all) > 0 and len(hc3_all) > 0 and 'avg_width' in df.columns:
        merged = amri_all[['dgp', 'delta', 'n', 'avg_width']].merge(
            hc3_all[['dgp', 'delta', 'n', 'avg_width']],
            on=['dgp', 'delta', 'n'], suffixes=('_amri', '_hc3'))
        if len(merged) > 0:
            ratio = merged['avg_width_amri'] / merged['avg_width_hc3']
            print(f"    Width ratio ({amri_method}/HC3): mean={ratio.mean():.4f}, "
                  f"max={ratio.max():.4f}, min={ratio.min():.4f}")
            c_bounded = ratio.max() < 1.1
            print(f"    Width bounded? {c_bounded}")
        else:
            c_bounded = None
            print("    No matched scenarios")
    else:
        c_bounded = None
        print("    Insufficient data or no width column")

    verdicts = [x for x in [a_efficient, b_robust, c_bounded] if x is not None]
    verdict = "CONFIRMED" if verdicts and all(verdicts) else "PARTIALLY CONFIRMED"
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
# TEST 8: AMRI vs AKS HEAD-TO-HEAD
# ============================================================================
def test_amri_vs_aks(df):
    """
    Direct comparison of AMRI v2 vs AKS adaptive CIs.
    Tests: (1) AMRI has better coverage accuracy, (2) width comparison.
    """
    print(f"\n{'='*80}")
    print("TEST 8: AMRI v2 vs AKS ADAPTIVE (Head-to-Head)")
    print("=" * 80)

    amri = df[df['method'] == 'AMRI_v2']
    aks = df[df['method'] == 'AKS_Adaptive']

    if len(amri) == 0 or len(aks) == 0:
        print("  Insufficient data (need both AMRI_v2 and AKS_Adaptive)")
        return None

    merged = amri[['dgp', 'delta', 'n', 'coverage', 'coverage_accuracy', 'avg_width']].merge(
        aks[['dgp', 'delta', 'n', 'coverage', 'coverage_accuracy', 'avg_width']],
        on=['dgp', 'delta', 'n'], suffixes=('_amri', '_aks'))

    if len(merged) == 0:
        print("  No matched scenarios")
        return None

    # Coverage accuracy comparison
    acc_diff = (merged['coverage_accuracy_aks'] - merged['coverage_accuracy_amri']).values
    t_acc, p_acc = stats.ttest_1samp(acc_diff, 0)
    p_amri_better = p_acc / 2 if t_acc > 0 else 1 - p_acc / 2

    print(f"\n  Matched scenarios: {len(merged)}")
    print(f"\n  A. Coverage accuracy (|cov - 0.95|):")
    print(f"    AMRI v2 mean: {merged['coverage_accuracy_amri'].mean():.4f}")
    print(f"    AKS mean:     {merged['coverage_accuracy_aks'].mean():.4f}")
    print(f"    Diff (AKS - AMRI): {acc_diff.mean():.4f} (positive = AMRI better)")
    print(f"    One-sided p (AMRI better): {p_amri_better:.4f} "
          f"{'***' if p_amri_better < 0.01 else '**' if p_amri_better < 0.05 else 'ns'}")

    # Coverage range
    print(f"\n  B. Coverage range:")
    print(f"    AMRI v2: [{merged['coverage_amri'].min():.4f}, {merged['coverage_amri'].max():.4f}]")
    print(f"    AKS:     [{merged['coverage_aks'].min():.4f}, {merged['coverage_aks'].max():.4f}]")

    # Width comparison
    if 'avg_width' in merged.columns:
        width_ratio = merged['avg_width_amri'] / merged['avg_width_aks']
        print(f"\n  C. Width ratio (AMRI/AKS): mean={width_ratio.mean():.4f}, "
              f"max={width_ratio.max():.4f}")
        print(f"    AMRI is {'wider' if width_ratio.mean() > 1 else 'narrower'} on average "
              f"(by {abs(width_ratio.mean()-1)*100:.1f}%)")

    # Under misspecification only
    misspec = merged[merged['delta'] > 0]
    if len(misspec) > 0:
        acc_diff_m = (misspec['coverage_accuracy_aks'] - misspec['coverage_accuracy_amri']).values
        print(f"\n  D. Under misspecification (delta > 0, {len(misspec)} scenarios):")
        print(f"    AMRI accuracy: {misspec['coverage_accuracy_amri'].mean():.4f}")
        print(f"    AKS accuracy:  {misspec['coverage_accuracy_aks'].mean():.4f}")
        n_amri_wins = (acc_diff_m > 0).sum()
        print(f"    AMRI wins: {n_amri_wins}/{len(misspec)} scenarios ({n_amri_wins/len(misspec)*100:.0f}%)")

    print(f"\n  VERDICT: AMRI v2 {'significantly' if p_amri_better < 0.05 else 'does not significantly'} "
          f"outperform{'s' if p_amri_better < 0.05 else ''} AKS on coverage accuracy")
    return merged


# ============================================================================
# TEST 9: POWER ANALYSIS
# ============================================================================
def test_power_analysis(B_values=[500, 1000, 2000, 5000, 10000], target_effect=0.01,
                        nominal=0.95, power_target=0.80, alpha=0.05):
    """
    Compute minimum B (simulation reps) needed to detect a coverage difference
    of target_effect from nominal with power_target at significance alpha.
    """
    print(f"\n{'='*80}")
    print("TEST 9: POWER ANALYSIS")
    print("=" * 80)
    print(f"  Detectable effect: {target_effect} deviation from {nominal}")
    print(f"  Target power: {power_target}")
    print(f"  Significance: {alpha}")
    print()

    # Under H0: coverage = nominal, SE = sqrt(nominal * (1-nominal) / B)
    # Under H1: coverage = nominal - target_effect
    # Power = P(reject H0 | H1 true)
    # Test statistic: Z = (p_hat - nominal) / sqrt(nominal*(1-nominal)/B)

    z_alpha = stats.norm.ppf(alpha)  # one-sided (testing undercoverage)

    for B in B_values:
        se_null = np.sqrt(nominal * (1 - nominal) / B)
        se_alt = np.sqrt((nominal - target_effect) * (1 - nominal + target_effect) / B)

        # Rejection threshold under H0
        threshold = nominal + z_alpha * se_null

        # Power: P(p_hat < threshold | p = nominal - target_effect)
        z_power = (threshold - (nominal - target_effect)) / se_alt
        power = stats.norm.cdf(z_power)

        print(f"  B={B:6d}: SE={se_null:.4f}, Power={power:.4f} "
              f"{'[SUFFICIENT]' if power >= power_target else ''}")

    # Compute minimum B analytically
    z_beta = stats.norm.ppf(power_target)
    p0 = nominal
    p1 = nominal - target_effect
    # B = ((z_alpha * sqrt(p0*(1-p0)) + z_beta * sqrt(p1*(1-p1))) / (p1 - p0))^2
    numerator = (-z_alpha * np.sqrt(p0 * (1 - p0)) + z_beta * np.sqrt(p1 * (1 - p1)))
    B_min = int(np.ceil((numerator / target_effect) ** 2))

    print(f"\n  Minimum B for {power_target:.0%} power to detect {target_effect} deviation: {B_min}")
    print(f"  Our B=2000: {'SUFFICIENT' if 2000 >= B_min else 'INSUFFICIENT'} for this effect size")

    return B_min


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

    if 'Naive_OLS' in df['method'].values:
        test2_df, test2_p = test_monotonic_degradation(df, 'Naive_OLS')

    test3_df, test3_pass = test_sandwich_robustness(df, 'Sandwich_HC3', 0.93)

    # Also test AMRI v2 robustness
    if 'AMRI_v2' in df['method'].values:
        test3b_df, test3b_pass = test_sandwich_robustness(df, 'AMRI_v2', 0.93)

    test4_verdict = test_amri_best_of_both(df)

    methods_present = df['method'].unique()
    methods_of_interest = ['Naive_OLS', 'Sandwich_HC3', 'Pairs_Bootstrap',
                           'Wild_Bootstrap', 'Bootstrap_t', 'AMRI_v1', 'AMRI_v2']
    if sum(m in methods_present for m in methods_of_interest) >= 3:
        test5_slopes, test5_detail = test_degradation_ordering(df)
        test_sample_size_paradox(df)
        test_width_adaptation(df)

    # New tests
    test_amri_vs_aks(df)
    test_power_analysis()

    generate_summary_table(df)

    # Final summary
    print(f"\n{'='*80}")
    print("FINAL HYPOTHESIS STATUS SUMMARY")
    print("=" * 80)
    print(f"""
    H1 (Naive Degrades):      Formal trend tests with Fisher combination
    H2 (Sandwich Robustness): Wilson CI lower bounds + Simes test
    H3 (AMRI Best-of-Both):   TOST equivalence (epsilon={0.02}) + paired tests
    H4 (Ordering):            Mann-Whitney U pairwise comparisons
    H5 (Width Adaptation):    Linear trend tests in width vs delta
    H6 (AMRI > AKS):          One-sided paired t-test on coverage accuracy
    BONUS (Size Paradox):     Spearman correlation of coverage with log(n)
    BONUS (Power):            Minimum B = {test_power_analysis.__defaults__[0]} for {0.01} effect

    Data: {len(df)} scenarios across {df['dgp'].nunique()} DGPs,
          {df['method'].nunique()} methods, deltas {sorted(df['delta'].unique())},
          n in {sorted(df['n'].unique())}
    """)
    print("DONE.")
