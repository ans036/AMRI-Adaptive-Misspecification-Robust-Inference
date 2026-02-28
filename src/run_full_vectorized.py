"""
FULL VECTORIZED SIMULATION - No shortcuts, no limits.
Uses closed-form OLS for ~10x speedup over statsmodels.
All DGPs x All severities x All sample sizes x All methods x 2000 reps.
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import warnings
import time

warnings.filterwarnings('ignore')

def log(msg):
    print(msg, flush=True)

# ============================================================================
# FAST CLOSED-FORM OLS
# ============================================================================

def fast_ols(X, Y):
    """Closed-form OLS: returns (beta, XtX_inv, residuals, sigma2)."""
    n = len(Y)
    Xa = np.column_stack([np.ones(n), X])
    XtX = Xa.T @ Xa
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ (Xa.T @ Y)
    resid = Y - Xa @ beta
    sigma2 = np.sum(resid**2) / (n - 2)
    return beta, XtX_inv, resid, sigma2, Xa


def naive_se(XtX_inv, sigma2):
    """Model-based SE for slope coefficient."""
    return np.sqrt(sigma2 * XtX_inv[1, 1])


def sandwich_se_hc0(Xa, XtX_inv, resid):
    """HC0 sandwich SE (White's original)."""
    n = len(resid)
    meat = Xa.T @ np.diag(resid**2) @ Xa
    V = XtX_inv @ meat @ XtX_inv
    return np.sqrt(V[1, 1])


def sandwich_se_hc3(Xa, XtX_inv, resid):
    """HC3 sandwich SE (jackknife-like correction)."""
    n = len(resid)
    h = np.sum(Xa * (Xa @ XtX_inv), axis=1)  # leverage values
    adjusted_resid2 = resid**2 / (1 - h)**2
    meat = Xa.T @ np.diag(adjusted_resid2) @ Xa
    V = XtX_inv @ meat @ XtX_inv
    return np.sqrt(V[1, 1])


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
# METHODS (all using fast closed-form OLS)
# ============================================================================

def method_naive(X, Y, alpha=0.05):
    beta, XtX_inv, resid, s2, Xa = fast_ols(X, Y)
    n = len(Y)
    theta = beta[1]
    se = naive_se(XtX_inv, s2)
    t_val = stats.t.ppf(1 - alpha/2, n - 2)
    return theta, se, theta - t_val * se, theta + t_val * se

def method_hc0(X, Y, alpha=0.05):
    beta, XtX_inv, resid, s2, Xa = fast_ols(X, Y)
    n = len(Y)
    theta = beta[1]
    se = sandwich_se_hc0(Xa, XtX_inv, resid)
    t_val = stats.t.ppf(1 - alpha/2, n - 2)
    return theta, se, theta - t_val * se, theta + t_val * se

def method_hc3(X, Y, alpha=0.05):
    beta, XtX_inv, resid, s2, Xa = fast_ols(X, Y)
    n = len(Y)
    theta = beta[1]
    se = sandwich_se_hc3(Xa, XtX_inv, resid)
    t_val = stats.t.ppf(1 - alpha/2, n - 2)
    return theta, se, theta - t_val * se, theta + t_val * se

def method_bayesian(X, Y, alpha=0.05):
    n = len(Y)
    Xa = np.column_stack([np.ones(n), X])
    prior_prec = np.eye(2) / 100.0
    XtX = Xa.T @ Xa
    post_var = np.linalg.inv(XtX + prior_prec)
    post_mean = post_var @ (Xa.T @ Y)
    resid = Y - Xa @ post_mean
    s2 = np.sum(resid**2) / (n - 2)
    theta = post_mean[1]
    se = np.sqrt(s2 * post_var[1, 1])
    t_val = stats.t.ppf(1 - alpha/2, n - 2)
    return theta, se, theta - t_val * se, theta + t_val * se

def method_amri(X, Y, alpha=0.05):
    beta, XtX_inv, resid, s2, Xa = fast_ols(X, Y)
    n = len(Y)
    theta = beta[1]
    se_naive = naive_se(XtX_inv, s2)
    se_robust = sandwich_se_hc3(Xa, XtX_inv, resid)
    ratio = se_robust / se_naive if se_naive > 1e-10 else 999
    threshold = 1 + 2 / np.sqrt(n)
    if ratio > threshold or ratio < 1 / threshold:
        se = se_robust * 1.05
    else:
        se = se_naive
    t_val = stats.t.ppf(1 - alpha/2, n - 2)
    return theta, se, theta - t_val * se, theta + t_val * se

def method_pairs_bootstrap(X, Y, alpha=0.05, B_boot=999):
    n = len(Y)
    beta, XtX_inv, resid, s2, Xa = fast_ols(X, Y)
    theta = beta[1]
    boot_rng = np.random.default_rng(42)
    boots = np.empty(B_boot)
    for b in range(B_boot):
        idx = boot_rng.integers(0, n, n)
        Xa_b = Xa[idx]
        Y_b = Y[idx]
        try:
            beta_b = np.linalg.solve(Xa_b.T @ Xa_b, Xa_b.T @ Y_b)
            boots[b] = beta_b[1]
        except:
            boots[b] = np.nan
    boots = boots[~np.isnan(boots)]
    if len(boots) < B_boot * 0.5:
        return np.nan, np.nan, np.nan, np.nan
    se = np.std(boots)
    return theta, se, np.percentile(boots, 2.5), np.percentile(boots, 97.5)

def method_wild_bootstrap(X, Y, alpha=0.05, B_boot=999):
    n = len(Y)
    beta, XtX_inv, resid, s2, Xa = fast_ols(X, Y)
    theta = beta[1]
    fitted = Xa @ beta
    boot_rng = np.random.default_rng(42)
    boots = np.empty(B_boot)
    XtX_solve = np.linalg.inv(Xa.T @ Xa)
    for b in range(B_boot):
        v = boot_rng.choice([-1.0, 1.0], size=n)
        Y_star = fitted + resid * v
        beta_b = XtX_solve @ (Xa.T @ Y_star)
        boots[b] = beta_b[1]
    se = np.std(boots)
    return theta, se, np.percentile(boots, 2.5), np.percentile(boots, 97.5)

def method_bootstrap_t(X, Y, alpha=0.05, B_boot=999):
    n = len(Y)
    beta, XtX_inv, resid, s2, Xa = fast_ols(X, Y)
    theta = beta[1]
    se_base = sandwich_se_hc3(Xa, XtX_inv, resid)
    boot_rng = np.random.default_rng(42)
    t_stars = []
    for b in range(B_boot):
        idx = boot_rng.integers(0, n, n)
        Xa_b = Xa[idx]
        Y_b = Y[idx]
        try:
            XtX_b = Xa_b.T @ Xa_b
            XtX_inv_b = np.linalg.inv(XtX_b)
            beta_b = XtX_inv_b @ (Xa_b.T @ Y_b)
            resid_b = Y_b - Xa_b @ beta_b
            se_b = sandwich_se_hc3(Xa_b, XtX_inv_b, resid_b)
            if se_b > 1e-10:
                t_stars.append((beta_b[1] - theta) / se_b)
        except:
            pass
    t_stars = np.array(t_stars)
    if len(t_stars) < B_boot * 0.3:
        return np.nan, np.nan, np.nan, np.nan
    q_lo, q_hi = np.percentile(t_stars, [2.5, 97.5])
    return theta, se_base, theta - q_hi * se_base, theta - q_lo * se_base

METHODS = {
    'Naive_OLS': method_naive,
    'Sandwich_HC0': method_hc0,
    'Sandwich_HC3': method_hc3,
    'Bayesian': method_bayesian,
    'AMRI': method_amri,
    'Pairs_Bootstrap': method_pairs_bootstrap,
    'Wild_Bootstrap': method_wild_bootstrap,
    'Bootstrap_t': method_bootstrap_t,
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
            if np.isnan(ci_lo) or np.isnan(ci_hi) or np.isinf(ci_lo) or np.isinf(ci_hi):
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
    biases = np.array(biases)
    return {
        'dgp': dgp_name, 'delta': delta, 'n': n, 'method': method_name,
        'B_total': B, 'B_valid': valid,
        'coverage': round(cov, 5),
        'coverage_mc_se': round(np.sqrt(cov * (1 - cov) / valid), 6),
        'avg_width': round(float(np.mean(widths)), 6),
        'median_width': round(float(np.median(widths)), 6),
        'bias': round(float(np.mean(biases)), 6),
        'rmse': round(float(np.sqrt(np.mean(biases**2))), 6),
        'se_mean': round(float(np.mean(se_hats)), 6),
    }

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    log("=" * 70)
    log("FULL COMPLETE SIMULATION - NO SHORTCUTS")
    log("6 DGPs x 5 severities x 6 sample sizes x 8 methods x 2000 reps")
    log("=" * 70)

    deltas = [0.0, 0.25, 0.5, 0.75, 1.0]
    sample_sizes = [50, 100, 250, 500, 1000, 5000]
    B_REPS = 2000

    scenarios = []
    for dgp_name in DGPS:
        for delta in deltas:
            for n in sample_sizes:
                for method_name in METHODS:
                    scenarios.append((dgp_name, delta, n, method_name))

    total = len(scenarios)
    log(f"Total scenarios: {total}")
    log(f"Total simulation runs: {total * B_REPS:,}")
    log(f"Methods: {list(METHODS.keys())}")
    log(f"DGPs: {list(DGPS.keys())}")
    log("")

    results = []
    t0 = time.time()

    for i, (dgp_name, delta, n, method_name) in enumerate(scenarios):
        ts = time.time()
        res = run_scenario(dgp_name, DGPS[dgp_name], delta, n,
                           method_name, METHODS[method_name], B=B_REPS)
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

        # Save intermediate results every 200 scenarios
        if (i + 1) % 200 == 0:
            pd.DataFrame(results).to_csv(
                "c:/Users/anish/OneDrive/Desktop/Novel Research/results/results_full_intermediate.csv",
                index=False)
            log(f"  >> Intermediate save: {len(results)} scenarios")

    df = pd.DataFrame(results)
    outpath = "c:/Users/anish/OneDrive/Desktop/Novel Research/results/results_full.csv"
    df.to_csv(outpath, index=False)

    total_time = time.time() - t0
    log(f"\n{'='*70}")
    log(f"SIMULATION COMPLETE")
    log(f"{'='*70}")
    log(f"Scenarios: {len(df)}/{total}")
    log(f"Total time: {total_time/60:.1f} min ({total_time/3600:.1f} hours)")
    log(f"Saved: {outpath}")

    # ===== COMPREHENSIVE RESULTS =====
    log(f"\n{'='*70}")
    log("OVERALL COVERAGE BY METHOD")
    log(f"{'='*70}")
    summary = df.groupby('method')['coverage'].agg(['mean', 'std', 'min', 'max', 'count'])
    log(summary.sort_values('mean', ascending=False).to_string(float_format=lambda x: f'{x:.4f}'))

    log(f"\n{'='*70}")
    log("COVERAGE: delta=0.0 (correct specification), n=500")
    log(f"{'='*70}")
    for method in sorted(df['method'].unique()):
        sub = df[(df['delta'] == 0.0) & (df['n'] == 500) & (df['method'] == method)]
        if len(sub) > 0:
            vals = sub['coverage'].values
            log(f"  {method:<20}: {vals} | mean={vals.mean():.4f}")

    log(f"\n{'='*70}")
    log("COVERAGE: delta=1.0 (severe misspecification), n=500")
    log(f"{'='*70}")
    for method in sorted(df['method'].unique()):
        sub = df[(df['delta'] == 1.0) & (df['n'] == 500) & (df['method'] == method)]
        if len(sub) > 0:
            vals = sub['coverage'].values
            log(f"  {method:<20}: {vals} | mean={vals.mean():.4f}")

    log(f"\n{'='*70}")
    log("COVERAGE DROP: mean(delta=0) - mean(delta=1) by method")
    log(f"{'='*70}")
    for method in sorted(df['method'].unique()):
        c0 = df[(df['delta'] == 0.0) & (df['n'] == 500) & (df['method'] == method)]['coverage'].mean()
        c1 = df[(df['delta'] == 1.0) & (df['n'] == 500) & (df['method'] == method)]['coverage'].mean()
        drop = c0 - c1
        log(f"  {method:<20}: {c0:.4f} -> {c1:.4f} (drop = {drop:.4f})")

    log(f"\n{'='*70}")
    log("AVERAGE INTERVAL WIDTH: delta=0.5, n=500")
    log(f"{'='*70}")
    for method in sorted(df['method'].unique()):
        sub = df[(df['delta'] == 0.5) & (df['n'] == 500) & (df['method'] == method)]
        if len(sub) > 0:
            vals = sub['avg_width'].values
            log(f"  {method:<20}: {vals} | mean={vals.mean():.4f}")

    log("\nDONE.")
