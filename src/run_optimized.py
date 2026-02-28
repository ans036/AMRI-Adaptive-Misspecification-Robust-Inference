"""
Optimized Simulation: Smart scheduling to finish in ~20-30 min.
- Fast methods (naive, sandwich): all sample sizes, 2000 reps
- Bootstrap methods: n <= 1000, 500 reps
- All 6 DGPs x 5 severity levels
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import warnings
import time
import os

warnings.filterwarnings('ignore')

def log(msg):
    print(msg, flush=True)

# ============================================================================
# DGPs
# ============================================================================
def dgp_nonlinearity(n, delta, rng):
    X = rng.standard_normal(n)
    Y = X + delta * X**2 + rng.standard_normal(n)
    return X, Y, 1.0

def dgp_heavy_tails(n, delta, rng):
    X = rng.standard_normal(n)
    if delta == 0:
        eps = rng.standard_normal(n)
    else:
        df = max(2.1, 10 - 8 * delta)
        eps = rng.standard_t(df, n) / np.sqrt(df / (df - 2))
    Y = X + eps
    return X, Y, 1.0

def dgp_heteroscedasticity(n, delta, rng):
    X = rng.standard_normal(n)
    sigma = np.exp(delta * X / 2)
    Y = X + rng.standard_normal(n) * sigma
    return X, Y, 1.0

def dgp_omitted_variable(n, delta, rng):
    rho = 0.5
    Z = rng.standard_normal((n, 2))
    X1 = Z[:, 0]
    X2 = rho * Z[:, 0] + np.sqrt(1 - rho**2) * Z[:, 1]
    Y = X1 + delta * X2 + rng.standard_normal(n)
    return X1, Y, 1.0 + delta * rho

def dgp_clustering(n, delta, rng):
    G = max(10, n // 10)
    m = n // G
    actual_n = G * m
    tau = np.sqrt(delta)
    u = rng.standard_normal(G) * tau
    X = rng.standard_normal(actual_n)
    cluster_ids = np.repeat(np.arange(G), m)
    Y = X + u[cluster_ids] + rng.standard_normal(actual_n)
    return X[:actual_n], Y[:actual_n], 1.0

def dgp_contaminated(n, delta, rng):
    X = rng.standard_normal(n)
    contam_prob = delta * 0.2
    is_outlier = rng.random(n) < contam_prob
    eps = np.where(is_outlier, rng.standard_normal(n) * 5.0, rng.standard_normal(n))
    Y = X + eps
    return X, Y, 1.0

DGPS = {
    'DGP1_nonlinearity': dgp_nonlinearity,
    'DGP2_heavy_tails': dgp_heavy_tails,
    'DGP3_heteroscedastic': dgp_heteroscedasticity,
    'DGP4_omitted_var': dgp_omitted_variable,
    'DGP5_clustering': dgp_clustering,
    'DGP6_contaminated': dgp_contaminated,
}

# ============================================================================
# METHODS
# ============================================================================
def m_naive(X, Y, alpha=0.05):
    Xa = sm.add_constant(X)
    m = sm.OLS(Y, Xa).fit()
    ci = m.conf_int(alpha=alpha)[1]
    return m.params[1], m.bse[1], ci[0], ci[1]

def m_hc0(X, Y, alpha=0.05):
    Xa = sm.add_constant(X)
    m = sm.OLS(Y, Xa).fit(cov_type='HC0')
    ci = m.conf_int(alpha=alpha)[1]
    return m.params[1], m.bse[1], ci[0], ci[1]

def m_hc3(X, Y, alpha=0.05):
    Xa = sm.add_constant(X)
    m = sm.OLS(Y, Xa).fit(cov_type='HC3')
    ci = m.conf_int(alpha=alpha)[1]
    return m.params[1], m.bse[1], ci[0], ci[1]

def m_bayes(X, Y, alpha=0.05):
    n = len(Y)
    Xa = sm.add_constant(X)
    prior_prec = np.eye(2) / 100.0
    XtX = Xa.T @ Xa
    post_var = np.linalg.inv(XtX + prior_prec)
    post_mean = post_var @ (Xa.T @ Y)
    resid = Y - Xa @ post_mean
    s2 = np.sum(resid**2) / (n - 2)
    theta = post_mean[1]
    se = np.sqrt(s2 * post_var[1, 1])
    t_val = stats.t.ppf(1 - alpha/2, n-2)
    return theta, se, theta - t_val * se, theta + t_val * se

def m_amri(X, Y, alpha=0.05):
    n = len(Y)
    Xa = sm.add_constant(X)
    naive = sm.OLS(Y, Xa).fit()
    robust = sm.OLS(Y, Xa).fit(cov_type='HC3')
    theta = naive.params[1]
    se_n = naive.bse[1]
    se_r = robust.bse[1]
    ratio = se_r / se_n if se_n > 1e-10 else 999
    threshold = 1 + 2 / np.sqrt(n)
    if ratio > threshold or ratio < 1 / threshold:
        se = se_r * 1.05
    else:
        se = se_n
    t_val = stats.t.ppf(1 - alpha/2, n-2)
    return theta, se, theta - t_val * se, theta + t_val * se

def m_pairs_boot(X, Y, alpha=0.05):
    n = len(Y)
    Xa = sm.add_constant(X)
    base = sm.OLS(Y, Xa).fit()
    theta = base.params[1]
    rng = np.random.default_rng(42)
    B = 199
    boots = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        boots[b] = sm.OLS(Y[idx], Xa[idx]).fit().params[1]
    se = np.std(boots)
    return theta, se, np.percentile(boots, 2.5), np.percentile(boots, 97.5)

def m_wild_boot(X, Y, alpha=0.05):
    n = len(Y)
    Xa = sm.add_constant(X)
    base = sm.OLS(Y, Xa).fit()
    theta = base.params[1]
    resid = base.resid
    fitted = base.fittedvalues
    rng = np.random.default_rng(42)
    B = 199
    boots = np.empty(B)
    for b in range(B):
        v = rng.choice([-1.0, 1.0], size=n)
        Y_star = fitted + resid * v
        boots[b] = sm.OLS(Y_star, Xa).fit().params[1]
    se = np.std(boots)
    return theta, se, np.percentile(boots, 2.5), np.percentile(boots, 97.5)

def m_boot_t(X, Y, alpha=0.05):
    n = len(Y)
    Xa = sm.add_constant(X)
    base = sm.OLS(Y, Xa).fit(cov_type='HC3')
    theta = base.params[1]
    se_base = base.bse[1]
    rng = np.random.default_rng(42)
    B = 199
    t_stars = []
    for b in range(B):
        idx = rng.integers(0, n, n)
        try:
            mm = sm.OLS(Y[idx], Xa[idx]).fit(cov_type='HC3')
            t_stars.append((mm.params[1] - theta) / mm.bse[1])
        except:
            pass
    t_stars = np.array(t_stars)
    if len(t_stars) < 50:
        return np.nan, np.nan, np.nan, np.nan
    q_lo, q_hi = np.percentile(t_stars, [2.5, 97.5])
    return theta, se_base, theta - q_hi * se_base, theta - q_lo * se_base

# Fast methods: run at all sample sizes with 2000 reps
FAST_METHODS = {
    'Naive_OLS': m_naive,
    'Sandwich_HC0': m_hc0,
    'Sandwich_HC3': m_hc3,
    'Bayesian': m_bayes,
    'AMRI': m_amri,
}

# Slow methods: run at n <= 1000 with 1000 reps
SLOW_METHODS = {
    'Pairs_Bootstrap': m_pairs_boot,
    'Wild_Bootstrap': m_wild_boot,
    'Bootstrap_t': m_boot_t,
}

# ============================================================================
# RUNNER
# ============================================================================
def run_scenario(dgp_name, dgp_func, delta, n, method_name, method_func, B):
    master = np.random.SeedSequence(20260228)
    seeds = master.spawn(B)
    covers = 0
    widths = []
    biases = []
    se_hats = []
    valid = 0

    for b in range(B):
        rng = np.random.default_rng(seeds[b])
        try:
            X, Y, theta_true = dgp_func(n, delta, rng)
            theta_hat, se_hat, ci_lo, ci_hi = method_func(X, Y)
            if np.isnan(ci_lo) or np.isnan(ci_hi):
                continue
            covers += int(ci_lo <= theta_true <= ci_hi)
            widths.append(ci_hi - ci_lo)
            biases.append(theta_hat - theta_true)
            se_hats.append(se_hat)
            valid += 1
        except:
            continue

    if valid < B * 0.3:
        return None

    cov = covers / valid
    return {
        'dgp': dgp_name, 'delta': delta, 'n': n, 'method': method_name,
        'B': valid,
        'coverage': round(cov, 4),
        'coverage_mc_se': round(np.sqrt(cov * (1 - cov) / valid), 5),
        'avg_width': round(np.mean(widths), 5),
        'median_width': round(np.median(widths), 5),
        'bias': round(np.mean(biases), 5),
        'rmse': round(np.sqrt(np.mean(np.array(biases)**2)), 5),
        'se_mean': round(np.mean(se_hats), 5),
    }


if __name__ == '__main__':
    log("=" * 70)
    log("OPTIMIZED SIMULATION")
    log("=" * 70)

    deltas = [0.0, 0.25, 0.5, 0.75, 1.0]
    sample_sizes_fast = [50, 100, 250, 500, 1000, 5000]
    sample_sizes_slow = [50, 100, 250, 500, 1000]  # skip n=5000 for bootstrap
    B_FAST = 2000
    B_SLOW = 1000

    # Build scenario list with priorities
    scenarios = []
    for dgp_name in DGPS:
        for delta in deltas:
            for n in sample_sizes_fast:
                for method_name, method_func in FAST_METHODS.items():
                    scenarios.append((dgp_name, delta, n, method_name, method_func, B_FAST))
            for n in sample_sizes_slow:
                for method_name, method_func in SLOW_METHODS.items():
                    scenarios.append((dgp_name, delta, n, method_name, method_func, B_SLOW))

    total = len(scenarios)
    total_runs = sum(s[5] for s in scenarios)
    log(f"Total scenarios: {total}")
    log(f"Total simulation runs: {total_runs:,}")
    log(f"Fast methods: {list(FAST_METHODS.keys())}")
    log(f"Slow methods: {list(SLOW_METHODS.keys())}")
    log("")

    results = []
    t0 = time.time()

    for i, (dgp_name, delta, n, method_name, method_func, B) in enumerate(scenarios):
        ts = time.time()
        res = run_scenario(dgp_name, DGPS[dgp_name], delta, n, method_name, method_func, B)
        te = time.time()
        if res is not None:
            results.append(res)

        if (i + 1) % 20 == 0 or i == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            pct = (i + 1) / total * 100
            log(f"  [{pct:5.1f}%] {i+1}/{total} | "
                f"{dgp_name[-15:]}, d={delta}, n={n:>5}, {method_name:<17} | "
                f"{te-ts:.1f}s | ETA: {eta/60:.0f}min")

    df = pd.DataFrame(results)
    outpath = "c:/Users/anish/OneDrive/Desktop/Novel Research/results/results_full.csv"
    df.to_csv(outpath, index=False)

    total_time = time.time() - t0
    log(f"\nDone! {len(df)} scenarios in {total_time/60:.1f} min")
    log(f"Saved: {outpath}")

    # ========================================
    # COMPREHENSIVE SUMMARY
    # ========================================
    log("\n" + "=" * 70)
    log("COVERAGE SUMMARY (all conditions)")
    log("=" * 70)
    summary = df.groupby('method')['coverage'].agg(['mean', 'std', 'min', 'max', 'count'])
    log(summary.sort_values('mean', ascending=False).to_string(float_format=lambda x: f'{x:.4f}'))

    log("\n" + "=" * 70)
    log("COVERAGE AT SEVERE MISSPECIFICATION (delta=1.0, n=500)")
    log("=" * 70)
    severe = df[(df['delta'] == 1.0) & (df['n'] == 500)]
    if len(severe) > 0:
        for method in sorted(severe['method'].unique()):
            msub = severe[severe['method'] == method]
            covs = msub['coverage'].values
            log(f"  {method:<20}: coverages={covs} | mean={covs.mean():.4f}")

    log("\n" + "=" * 70)
    log("COVERAGE AT NO MISSPECIFICATION (delta=0.0, n=500)")
    log("=" * 70)
    correct = df[(df['delta'] == 0.0) & (df['n'] == 500)]
    if len(correct) > 0:
        for method in sorted(correct['method'].unique()):
            msub = correct[correct['method'] == method]
            covs = msub['coverage'].values
            log(f"  {method:<20}: coverages={covs} | mean={covs.mean():.4f}")

    log("\n" + "=" * 70)
    log("INTERVAL WIDTH AT delta=0.5, n=500")
    log("=" * 70)
    moderate = df[(df['delta'] == 0.5) & (df['n'] == 500)]
    if len(moderate) > 0:
        for method in sorted(moderate['method'].unique()):
            msub = moderate[moderate['method'] == method]
            widths = msub['avg_width'].values
            log(f"  {method:<20}: avg_widths={widths} | mean={widths.mean():.4f}")
