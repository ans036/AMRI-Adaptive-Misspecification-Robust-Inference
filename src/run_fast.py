"""
Optimized Fast Simulation: All DGPs, core methods, reduced bootstrap reps.
Targets ~30-45 min total runtime.
"""
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import warnings
import time
import os

warnings.filterwarnings('ignore')

# ============================================================================
# DGPs (same as simulation.py)
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

def dgp_contaminated_normal(n, delta, rng):
    """DGP7: Contaminated normal (mixture model misspecification)."""
    X = rng.standard_normal(n)
    # With probability delta*0.2, error comes from N(0, 25); otherwise N(0, 1)
    contam_prob = delta * 0.2  # 0% to 20% contamination
    is_outlier = rng.random(n) < contam_prob
    eps = np.where(is_outlier, rng.standard_normal(n) * 5.0, rng.standard_normal(n))
    Y = X + eps
    return X, Y, 1.0

DGPS = {
    'DGP1_nonlinearity': dgp_nonlinearity,
    'DGP2_heavy_tails': dgp_heavy_tails,
    'DGP3_heteroscedasticity': dgp_heteroscedasticity,
    'DGP4_omitted_variable': dgp_omitted_variable,
    'DGP5_clustering': dgp_clustering,
    'DGP6_contaminated': dgp_contaminated_normal,
}

# ============================================================================
# METHODS (optimized for speed)
# ============================================================================

def method_naive(X, Y, alpha=0.05):
    Xa = sm.add_constant(X)
    model = sm.OLS(Y, Xa).fit()
    ci = model.conf_int(alpha=alpha)[1]
    return model.params[1], model.bse[1], ci[0], ci[1]

def method_hc0(X, Y, alpha=0.05):
    Xa = sm.add_constant(X)
    model = sm.OLS(Y, Xa).fit(cov_type='HC0')
    ci = model.conf_int(alpha=alpha)[1]
    return model.params[1], model.bse[1], ci[0], ci[1]

def method_hc3(X, Y, alpha=0.05):
    Xa = sm.add_constant(X)
    model = sm.OLS(Y, Xa).fit(cov_type='HC3')
    ci = model.conf_int(alpha=alpha)[1]
    return model.params[1], model.bse[1], ci[0], ci[1]

def method_pairs_boot(X, Y, alpha=0.05, B=299):
    n = len(Y)
    Xa = sm.add_constant(X)
    base = sm.OLS(Y, Xa).fit()
    theta = base.params[1]
    rng = np.random.default_rng(42)
    boots = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        boots[b] = sm.OLS(Y[idx], Xa[idx]).fit().params[1]
    se = np.std(boots)
    return theta, se, np.percentile(boots, 2.5), np.percentile(boots, 97.5)

def method_wild_boot(X, Y, alpha=0.05, B=299):
    n = len(Y)
    Xa = sm.add_constant(X)
    base = sm.OLS(Y, Xa).fit()
    theta = base.params[1]
    resid = base.resid
    fitted = base.fittedvalues
    rng = np.random.default_rng(42)
    boots = np.empty(B)
    for b in range(B):
        v = rng.choice([-1.0, 1.0], size=n)
        Y_star = fitted + resid * v
        boots[b] = sm.OLS(Y_star, Xa).fit().params[1]
    se = np.std(boots)
    return theta, se, np.percentile(boots, 2.5), np.percentile(boots, 97.5)

def method_boot_t(X, Y, alpha=0.05, B=299):
    n = len(Y)
    Xa = sm.add_constant(X)
    base = sm.OLS(Y, Xa).fit(cov_type='HC3')
    theta = base.params[1]
    se_base = base.bse[1]
    rng = np.random.default_rng(42)
    t_stars = []
    for b in range(B):
        idx = rng.integers(0, n, n)
        try:
            m = sm.OLS(Y[idx], Xa[idx]).fit(cov_type='HC3')
            t_stars.append((m.params[1] - theta) / m.bse[1])
        except Exception:
            pass
    t_stars = np.array(t_stars)
    q_lo, q_hi = np.percentile(t_stars, [2.5, 97.5])
    return theta, se_base, theta - q_hi * se_base, theta - q_lo * se_base

def method_bayesian(X, Y, alpha=0.05):
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
    return theta, se, theta - stats.t.ppf(0.975, n-2) * se, theta + stats.t.ppf(0.975, n-2) * se

def method_amri(X, Y, alpha=0.05):
    n = len(Y)
    Xa = sm.add_constant(X)
    naive = sm.OLS(Y, Xa).fit()
    robust = sm.OLS(Y, Xa).fit(cov_type='HC3')
    theta = naive.params[1]
    se_n = naive.bse[1]
    se_r = robust.bse[1]
    ratio = se_r / se_n if se_n > 0 else 999
    threshold = 1 + 2 / np.sqrt(n)
    if ratio > threshold or ratio < 1 / threshold:
        se = se_r * 1.05
    else:
        se = se_n
    return theta, se, theta - stats.t.ppf(0.975, n-2) * se, theta + stats.t.ppf(0.975, n-2) * se

METHODS = {
    'Naive_OLS': method_naive,
    'Sandwich_HC0': method_hc0,
    'Sandwich_HC3': method_hc3,
    'Pairs_Bootstrap': method_pairs_boot,
    'Wild_Bootstrap': method_wild_boot,
    'Bootstrap_t': method_boot_t,
    'Bayesian_Normal': method_bayesian,
    'AMRI': method_amri,
}

# ============================================================================
# RUNNER
# ============================================================================

def run_scenario(dgp_name, dgp_func, delta, n, method_name, method_func, B=1000):
    """Run B reps of one scenario."""
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
        except Exception:
            continue

    if valid < B * 0.5:
        return None

    cov = covers / valid
    return {
        'dgp': dgp_name, 'delta': delta, 'n': n, 'method': method_name,
        'B': valid,
        'coverage': cov,
        'coverage_mc_se': np.sqrt(cov * (1 - cov) / valid),
        'avg_width': np.mean(widths),
        'median_width': np.median(widths),
        'bias': np.mean(biases),
        'rmse': np.sqrt(np.mean(np.array(biases)**2)),
        'se_mean': np.mean(se_hats),
    }


if __name__ == '__main__':
    print("=" * 70)
    print("FAST COMPREHENSIVE SIMULATION")
    print("6 DGPs × 5 severities × 6 sample sizes × 8 methods × 1500 reps")
    print("=" * 70)

    deltas = [0.0, 0.25, 0.5, 0.75, 1.0]
    sample_sizes = [50, 100, 250, 500, 1000, 5000]
    B_REPS = 1500

    # Build scenario list
    scenarios = []
    for dgp_name in DGPS:
        for delta in deltas:
            for n in sample_sizes:
                for method_name in METHODS:
                    scenarios.append((dgp_name, delta, n, method_name))

    total = len(scenarios)
    print(f"Total scenarios: {total}")
    print(f"Total runs: {total * B_REPS:,}")
    print()

    results = []
    t0 = time.time()

    for i, (dgp_name, delta, n, method_name) in enumerate(scenarios):
        ts = time.time()
        res = run_scenario(dgp_name, DGPS[dgp_name], delta, n,
                           method_name, METHODS[method_name], B=B_REPS)
        te = time.time()
        if res is not None:
            results.append(res)

        if (i + 1) % 50 == 0 or i == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            pct = (i + 1) / total * 100
            print(f"  [{pct:5.1f}%] {i+1}/{total} | "
                  f"{dgp_name[-15:]}, d={delta}, n={n:>5}, {method_name:<17} | "
                  f"{te-ts:.1f}s | ETA: {eta/60:.0f}min")

    df = pd.DataFrame(results)
    outpath = "c:/Users/anish/OneDrive/Desktop/Novel Research/results/results_full.csv"
    df.to_csv(outpath, index=False)

    total_time = time.time() - t0
    print(f"\nDone! {len(df)} scenarios completed in {total_time/60:.1f} min")
    print(f"Saved to: {outpath}")

    # Quick coverage summary
    print("\n" + "=" * 70)
    print("COVERAGE SUMMARY (averaged across all DGPs, deltas, sample sizes)")
    print("=" * 70)
    summary = df.groupby('method')['coverage'].agg(['mean', 'std', 'min', 'max'])
    print(summary.sort_values('mean', ascending=False).to_string(float_format='%.4f'))

    print("\n" + "=" * 70)
    print("COVERAGE AT delta=1.0, n=500 (severe misspecification)")
    print("=" * 70)
    severe = df[(df['delta'] == 1.0) & (df['n'] == 500)]
    for method in sorted(severe['method'].unique()):
        msub = severe[severe['method'] == method]
        print(f"  {method:<20}: {msub['coverage'].values} | mean={msub['coverage'].mean():.4f}")
