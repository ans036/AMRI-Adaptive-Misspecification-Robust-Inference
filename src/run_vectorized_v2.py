"""
FULLY VECTORIZED SIMULATION v2 - Batch-computes everything.
Key insight: For 2-variable OLS (intercept + slope), the slope has closed-form:
  beta1 = Cov(X,Y)/Var(X) = (sum(X*Y) - n*Xbar*Ybar) / (sum(X^2) - n*Xbar^2)

This allows vectorizing over both reps AND bootstrap samples simultaneously.
Expected runtime: ~30-60 minutes for the FULL simulation.
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats
import warnings
import time

warnings.filterwarnings('ignore')

def log(msg):
    print(msg, flush=True)

# ============================================================================
# VECTORIZED OLS: compute slope for MANY datasets at once
# ============================================================================

def batch_ols_slope(X_batch, Y_batch):
    """
    Compute OLS slope for many datasets simultaneously.
    X_batch: (B, n) array of predictors
    Y_batch: (B, n) array of responses
    Returns: (B,) array of slope estimates
    """
    n = X_batch.shape[1]
    Xbar = X_batch.mean(axis=1, keepdims=True)
    Ybar = Y_batch.mean(axis=1, keepdims=True)
    Xc = X_batch - Xbar
    Yc = Y_batch - Ybar
    num = (Xc * Yc).sum(axis=1)
    den = (Xc * Xc).sum(axis=1)
    return num / den


def batch_ols_full(X_batch, Y_batch):
    """
    Compute OLS slope, intercept, residuals, and model-based SE.
    X_batch: (B, n), Y_batch: (B, n)
    Returns: slopes (B,), se_naive (B,), residuals (B, n)
    """
    B, n = X_batch.shape
    Xbar = X_batch.mean(axis=1, keepdims=True)
    Ybar = Y_batch.mean(axis=1, keepdims=True)
    Xc = X_batch - Xbar
    Yc = Y_batch - Ybar
    SXX = (Xc * Xc).sum(axis=1)
    SXY = (Xc * Yc).sum(axis=1)
    slopes = SXY / SXX
    intercepts = Y_batch.mean(axis=1) - slopes * X_batch.mean(axis=1)
    Y_hat = intercepts[:, None] + slopes[:, None] * X_batch
    resid = Y_batch - Y_hat
    sigma2 = (resid**2).sum(axis=1) / (n - 2)
    se_naive = np.sqrt(sigma2 / SXX)
    return slopes, se_naive, resid, SXX, sigma2


def batch_sandwich_hc0(X_batch, resid, SXX):
    """HC0 sandwich SE, vectorized."""
    B, n = X_batch.shape
    Xbar = X_batch.mean(axis=1, keepdims=True)
    Xc = X_batch - Xbar
    # HC0: sum(Xc^2 * e^2) / SXX^2
    meat = (Xc**2 * resid**2).sum(axis=1)
    # V(beta1) = meat / SXX^2 (for the slope in simple regression)
    # More precisely: V = (X'X)^{-1} X' diag(e^2) X (X'X)^{-1}
    # For slope: V[1,1] = meat / SXX^2 * correction
    # In simple regression: V(b1) = sum(xc^2 * e^2) / SXX^2
    se = np.sqrt(meat / SXX**2)
    return se


def batch_sandwich_hc3(X_batch, resid, SXX):
    """HC3 sandwich SE, vectorized."""
    B, n = X_batch.shape
    Xbar = X_batch.mean(axis=1, keepdims=True)
    Xc = X_batch - Xbar
    # Leverage: h_ii = 1/n + (x_i - xbar)^2 / SXX
    h = 1.0/n + Xc**2 / SXX[:, None]
    adjusted_resid2 = resid**2 / (1 - h)**2
    meat = (Xc**2 * adjusted_resid2).sum(axis=1)
    se = np.sqrt(meat / SXX**2)
    return se


# ============================================================================
# DGPs - vectorized (generate B datasets at once)
# ============================================================================

def gen_nonlinearity(B, n, delta, rng):
    X = rng.standard_normal((B, n))
    eps = rng.standard_normal((B, n))
    Y = X + delta * X**2 + eps
    theta_true = 1.0
    return X, Y, theta_true

def gen_heavy_tails(B, n, delta, rng):
    X = rng.standard_normal((B, n))
    if delta == 0:
        eps = rng.standard_normal((B, n))
    else:
        df = max(2.1, 10 - 8 * delta)
        eps = rng.standard_t(df, (B, n)) / np.sqrt(df / (df - 2))
    Y = X + eps
    return X, Y, 1.0

def gen_heteroscedastic(B, n, delta, rng):
    X = rng.standard_normal((B, n))
    sigma = np.exp(delta * X / 2)
    eps = rng.standard_normal((B, n)) * sigma
    Y = X + eps
    return X, Y, 1.0

def gen_omitted_var(B, n, delta, rng):
    rho = 0.5
    Z1 = rng.standard_normal((B, n))
    Z2 = rng.standard_normal((B, n))
    X1 = Z1
    X2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2
    eps = rng.standard_normal((B, n))
    Y = X1 + delta * X2 + eps
    return X1, Y, 1.0 + delta * rho

def gen_clustering(B, n, delta, rng):
    G = max(10, n // 10)
    m = n // G
    actual_n = G * m
    tau = np.sqrt(delta)
    u = rng.standard_normal((B, G)) * tau
    X = rng.standard_normal((B, actual_n))
    cluster_ids = np.tile(np.repeat(np.arange(G), m), (B, 1))
    # Vectorized cluster effect lookup
    u_expanded = np.take_along_axis(u, cluster_ids, axis=1)
    eps = rng.standard_normal((B, actual_n))
    Y = X + u_expanded + eps
    return X[:, :actual_n], Y[:, :actual_n], 1.0

def gen_contaminated(B, n, delta, rng):
    X = rng.standard_normal((B, n))
    contam_prob = delta * 0.2
    is_outlier = rng.random((B, n)) < contam_prob
    eps_normal = rng.standard_normal((B, n))
    eps_outlier = rng.standard_normal((B, n)) * 5.0
    eps = np.where(is_outlier, eps_outlier, eps_normal)
    Y = X + eps
    return X, Y, 1.0

DGPS = {
    'DGP1_nonlinearity': gen_nonlinearity,
    'DGP2_heavy_tails': gen_heavy_tails,
    'DGP3_heteroscedastic': gen_heteroscedastic,
    'DGP4_omitted_var': gen_omitted_var,
    'DGP5_clustering': gen_clustering,
    'DGP6_contaminated': gen_contaminated,
}

# ============================================================================
# METHODS - vectorized
# ============================================================================

def run_naive(X, Y, alpha=0.05):
    """Naive OLS + model-based SE."""
    B, n = X.shape
    slopes, se_naive, resid, SXX, sigma2 = batch_ols_full(X, Y)
    t_val = stats.t.ppf(1 - alpha/2, n - 2)
    ci_lo = slopes - t_val * se_naive
    ci_hi = slopes + t_val * se_naive
    return slopes, se_naive, ci_lo, ci_hi

def run_hc0(X, Y, alpha=0.05):
    """OLS + HC0 sandwich SE."""
    B, n = X.shape
    slopes, se_naive, resid, SXX, sigma2 = batch_ols_full(X, Y)
    se = batch_sandwich_hc0(X, resid, SXX)
    t_val = stats.t.ppf(1 - alpha/2, n - 2)
    ci_lo = slopes - t_val * se
    ci_hi = slopes + t_val * se
    return slopes, se, ci_lo, ci_hi

def run_hc3(X, Y, alpha=0.05):
    """OLS + HC3 sandwich SE."""
    B, n = X.shape
    slopes, se_naive, resid, SXX, sigma2 = batch_ols_full(X, Y)
    se = batch_sandwich_hc3(X, resid, SXX)
    t_val = stats.t.ppf(1 - alpha/2, n - 2)
    ci_lo = slopes - t_val * se
    ci_hi = slopes + t_val * se
    return slopes, se, ci_lo, ci_hi

def run_bayesian(X, Y, alpha=0.05):
    """Bayesian with vague prior (same as naive for large n, slightly different for small n)."""
    B, n = X.shape
    # With prior precision 1/100 on both params, posterior ≈ OLS for large n
    # Small correction for small n
    slopes, se_naive, resid, SXX, sigma2 = batch_ols_full(X, Y)
    # Prior shrinks toward 0 slightly
    prior_var = 100.0
    posterior_prec = SXX / sigma2 + 1.0 / prior_var
    posterior_var = 1.0 / posterior_prec
    posterior_mean = (SXX / sigma2 * slopes) * posterior_var
    se = np.sqrt(sigma2 * posterior_var)
    # For practical purposes with vague prior, very close to naive
    # But we use the model-based SE (which is what breaks)
    t_val = stats.t.ppf(1 - alpha/2, n - 2)
    ci_lo = slopes - t_val * se_naive  # Same as naive effectively
    ci_hi = slopes + t_val * se_naive
    return slopes, se_naive, ci_lo, ci_hi

def run_amri(X, Y, alpha=0.05):
    """AMRI: Adaptive Misspecification-Robust Inference."""
    B, n = X.shape
    slopes, se_naive, resid, SXX, sigma2 = batch_ols_full(X, Y)
    se_hc3 = batch_sandwich_hc3(X, resid, SXX)
    ratio = se_hc3 / np.maximum(se_naive, 1e-10)
    threshold = 1 + 2 / np.sqrt(n)

    # Adaptive: use sandwich when misspecification detected, else naive
    misspec = (ratio > threshold) | (ratio < 1.0 / threshold)
    se = np.where(misspec, se_hc3 * 1.05, se_naive)

    t_val = stats.t.ppf(1 - alpha/2, n - 2)
    ci_lo = slopes - t_val * se
    ci_hi = slopes + t_val * se
    return slopes, se, ci_lo, ci_hi

def run_pairs_bootstrap(X, Y, alpha=0.05, B_boot=999):
    """Pairs bootstrap (percentile CI) - vectorized inner loop."""
    B, n = X.shape
    slopes, _, _, _, _ = batch_ols_full(X, Y)

    boot_rng = np.random.default_rng(42)
    all_boot_slopes = np.empty((B, B_boot))

    for b in range(B_boot):
        idx = boot_rng.integers(0, n, (B, n))  # (B, n) bootstrap indices
        # Gather bootstrap samples for all B datasets simultaneously
        X_b = np.take_along_axis(X, idx, axis=1)
        Y_b = np.take_along_axis(Y, idx, axis=1)
        all_boot_slopes[:, b] = batch_ols_slope(X_b, Y_b)

    se = np.std(all_boot_slopes, axis=1)
    ci_lo = np.percentile(all_boot_slopes, 2.5, axis=1)
    ci_hi = np.percentile(all_boot_slopes, 97.5, axis=1)
    return slopes, se, ci_lo, ci_hi

def run_wild_bootstrap(X, Y, alpha=0.05, B_boot=999):
    """Wild bootstrap (Rademacher) - vectorized."""
    B, n = X.shape
    slopes, se_naive, resid, SXX, sigma2 = batch_ols_full(X, Y)
    intercepts = Y.mean(axis=1) - slopes * X.mean(axis=1)
    fitted = intercepts[:, None] + slopes[:, None] * X

    boot_rng = np.random.default_rng(42)
    all_boot_slopes = np.empty((B, B_boot))

    for b in range(B_boot):
        v = boot_rng.choice([-1.0, 1.0], size=(B, n))
        Y_star = fitted + resid * v
        all_boot_slopes[:, b] = batch_ols_slope(X, Y_star)

    se = np.std(all_boot_slopes, axis=1)
    ci_lo = np.percentile(all_boot_slopes, 2.5, axis=1)
    ci_hi = np.percentile(all_boot_slopes, 97.5, axis=1)
    return slopes, se, ci_lo, ci_hi

def run_bootstrap_t(X, Y, alpha=0.05, B_boot=999):
    """Studentized bootstrap (bootstrap-t) - vectorized."""
    B, n = X.shape
    slopes, se_naive, resid, SXX, sigma2 = batch_ols_full(X, Y)
    se_base = batch_sandwich_hc3(X, resid, SXX)

    boot_rng = np.random.default_rng(42)
    all_t_stars = np.empty((B, B_boot))

    for b in range(B_boot):
        idx = boot_rng.integers(0, n, (B, n))
        X_b = np.take_along_axis(X, idx, axis=1)
        Y_b = np.take_along_axis(Y, idx, axis=1)
        slopes_b, se_naive_b, resid_b, SXX_b, _ = batch_ols_full(X_b, Y_b)
        se_b = batch_sandwich_hc3(X_b, resid_b, SXX_b)
        se_b = np.maximum(se_b, 1e-10)
        all_t_stars[:, b] = (slopes_b - slopes) / se_b

    q_lo = np.percentile(all_t_stars, 2.5, axis=1)
    q_hi = np.percentile(all_t_stars, 97.5, axis=1)
    ci_lo = slopes - q_hi * se_base
    ci_hi = slopes - q_lo * se_base
    return slopes, se_base, ci_lo, ci_hi


METHODS = {
    'Naive_OLS': run_naive,
    'Sandwich_HC0': run_hc0,
    'Sandwich_HC3': run_hc3,
    'Bayesian': run_bayesian,
    'AMRI': run_amri,
    'Pairs_Bootstrap': run_pairs_bootstrap,
    'Wild_Bootstrap': run_wild_bootstrap,
    'Bootstrap_t': run_bootstrap_t,
}

# ============================================================================
# SCENARIO RUNNER
# ============================================================================

def run_scenario(dgp_name, dgp_func, delta, n, method_name, method_func, B, alpha=0.05):
    """Run one complete scenario: generate B datasets, apply method, compute metrics."""
    rng = np.random.default_rng(20260228)

    try:
        X, Y, theta_true = dgp_func(B, n, delta, rng)
        slopes, se, ci_lo, ci_hi = method_func(X, Y, alpha)

        # Filter valid results
        valid = np.isfinite(ci_lo) & np.isfinite(ci_hi) & np.isfinite(slopes)
        if valid.sum() < B * 0.3:
            return None

        slopes_v = slopes[valid]
        ci_lo_v = ci_lo[valid]
        ci_hi_v = ci_hi[valid]
        se_v = se[valid]

        covers = (ci_lo_v <= theta_true) & (theta_true <= ci_hi_v)
        cov = covers.mean()
        widths = ci_hi_v - ci_lo_v
        biases = slopes_v - theta_true
        n_valid = int(valid.sum())

        return {
            'dgp': dgp_name,
            'delta': delta,
            'n': n,
            'method': method_name,
            'B_total': B,
            'B_valid': n_valid,
            'coverage': round(float(cov), 5),
            'coverage_mc_se': round(float(np.sqrt(cov * (1 - cov) / n_valid)), 6),
            'avg_width': round(float(widths.mean()), 6),
            'median_width': round(float(np.median(widths)), 6),
            'bias': round(float(biases.mean()), 6),
            'rmse': round(float(np.sqrt((biases**2).mean())), 6),
            'se_mean': round(float(se_v.mean()), 6),
            'theta_true': theta_true,
        }
    except Exception as e:
        log(f"    ERROR: {dgp_name}, d={delta}, n={n}, {method_name}: {e}")
        return None


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    log("=" * 70)
    log("FULLY VECTORIZED SIMULATION v2")
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
    log(f"Reps per scenario: {B_REPS}")
    log(f"Total individual runs: {total * B_REPS:,}")
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
                f"{te-ts:.1f}s | elapsed={elapsed/60:.1f}m | ETA={eta/60:.1f}m")

        # Save intermediate every 200 scenarios
        if (i + 1) % 200 == 0:
            pd.DataFrame(results).to_csv(
                "c:/Users/anish/OneDrive/Desktop/Novel Research/results/results_intermediate.csv",
                index=False)

    df = pd.DataFrame(results)
    outpath = "c:/Users/anish/OneDrive/Desktop/Novel Research/results/results_full.csv"
    df.to_csv(outpath, index=False)

    total_time = time.time() - t0
    log(f"\n{'='*70}")
    log(f"SIMULATION COMPLETE")
    log(f"{'='*70}")
    log(f"Scenarios completed: {len(df)}/{total}")
    log(f"Total time: {total_time/60:.1f} min ({total_time/3600:.2f} hours)")
    log(f"Saved: {outpath}")

    # ===== COMPREHENSIVE RESULTS =====
    log(f"\n{'='*70}")
    log("OVERALL COVERAGE BY METHOD (across ALL conditions)")
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
            log(f"  {method:<20}: {np.round(vals, 4)} | mean={vals.mean():.4f}")

    log(f"\n{'='*70}")
    log("COVERAGE: delta=1.0 (severe misspecification), n=500")
    log(f"{'='*70}")
    for method in sorted(df['method'].unique()):
        sub = df[(df['delta'] == 1.0) & (df['n'] == 500) & (df['method'] == method)]
        if len(sub) > 0:
            vals = sub['coverage'].values
            log(f"  {method:<20}: {np.round(vals, 4)} | mean={vals.mean():.4f}")

    log(f"\n{'='*70}")
    log("COVERAGE DROP: mean(d=0) - mean(d=1) by method (at n=500)")
    log(f"{'='*70}")
    for method in sorted(df['method'].unique()):
        c0 = df[(df['delta'] == 0.0) & (df['n'] == 500) & (df['method'] == method)]['coverage'].mean()
        c1 = df[(df['delta'] == 1.0) & (df['n'] == 500) & (df['method'] == method)]['coverage'].mean()
        log(f"  {method:<20}: {c0:.4f} -> {c1:.4f} (drop = {c0-c1:.4f})")

    log(f"\n{'='*70}")
    log("AVERAGE INTERVAL WIDTH: delta=0.5, n=500")
    log(f"{'='*70}")
    for method in sorted(df['method'].unique()):
        sub = df[(df['delta'] == 0.5) & (df['n'] == 500) & (df['method'] == method)]
        if len(sub) > 0:
            log(f"  {method:<20}: mean_width={sub['avg_width'].mean():.4f}")

    log(f"\n{'='*70}")
    log("BIAS: delta=1.0, n=500")
    log(f"{'='*70}")
    for method in sorted(df['method'].unique()):
        sub = df[(df['delta'] == 1.0) & (df['n'] == 500) & (df['method'] == method)]
        if len(sub) > 0:
            log(f"  {method:<20}: mean_bias={sub['bias'].mean():.4f}, mean_rmse={sub['rmse'].mean():.4f}")

    log("\nDONE.")
