"""
Standalone AMRI v2 Simulation
=============================
Runs AMRI v2 (soft-thresholding) across ALL scenarios matching the full simulation.
Since AMRI v2 is non-bootstrap (just OLS + HC3 + blending), this runs very fast.

Results are saved separately and can be merged with the full simulation CSV.
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import time
import warnings

warnings.filterwarnings('ignore')

def log(msg):
    print(msg, flush=True)

# ============================================================================
# VECTORIZED OLS (copied from run_vectorized_v2.py for standalone use)
# ============================================================================

def batch_ols_full(X_batch, Y_batch):
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


def batch_sandwich_hc3(X_batch, resid, SXX):
    B, n = X_batch.shape
    Xbar = X_batch.mean(axis=1, keepdims=True)
    Xc = X_batch - Xbar
    h = 1.0/n + Xc**2 / SXX[:, None]
    adjusted_resid2 = resid**2 / (1 - h)**2
    meat = (Xc**2 * adjusted_resid2).sum(axis=1)
    se = np.sqrt(meat / SXX**2)
    return se


# ============================================================================
# DGPs (same as run_vectorized_v2.py)
# ============================================================================

def gen_nonlinearity(B, n, delta, rng):
    X = rng.standard_normal((B, n))
    eps = rng.standard_normal((B, n))
    Y = X + delta * X**2 + eps
    return X, Y, 1.0

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
# AMRI v1 + v2 METHODS
# ============================================================================

def run_amri_v1(X, Y, alpha=0.05):
    """AMRI v1: Hard switching."""
    B, n = X.shape
    slopes, se_naive, resid, SXX, sigma2 = batch_ols_full(X, Y)
    se_hc3 = batch_sandwich_hc3(X, resid, SXX)
    ratio = se_hc3 / np.maximum(se_naive, 1e-10)
    threshold = 1 + 2 / np.sqrt(n)
    misspec = (ratio > threshold) | (ratio < 1.0 / threshold)
    se = np.where(misspec, se_hc3 * 1.05, se_naive)
    t_val = stats.t.ppf(1 - alpha/2, n - 2)
    ci_lo = slopes - t_val * se
    ci_hi = slopes + t_val * se
    return slopes, se, ci_lo, ci_hi


def run_amri_v2(X, Y, alpha=0.05, c1=1.0, c2=2.0):
    """AMRI v2: Soft-thresholding (continuous, no pre-test discontinuity)."""
    B, n = X.shape
    slopes, se_naive, resid, SXX, sigma2 = batch_ols_full(X, Y)
    se_hc3 = batch_sandwich_hc3(X, resid, SXX)

    ratio = se_hc3 / np.maximum(se_naive, 1e-10)
    log_ratio = np.abs(np.log(ratio))

    lower = c1 / np.sqrt(n)
    upper = c2 / np.sqrt(n)
    w = np.clip((log_ratio - lower) / (upper - lower), 0.0, 1.0)

    se = (1 - w) * se_naive + w * se_hc3

    t_val = stats.t.ppf(1 - alpha/2, n - 2)
    ci_lo = slopes - t_val * se
    ci_hi = slopes + t_val * se
    return slopes, se, ci_lo, ci_hi, w


# ============================================================================
# SCENARIO RUNNER
# ============================================================================

def run_scenario(dgp_name, dgp_func, delta, n, method_name, method_func, B, alpha=0.05):
    """Run one complete scenario."""
    rng = np.random.default_rng(20260228)  # Same seed as full simulation

    try:
        X, Y, theta_true = dgp_func(B, n, delta, rng)

        if method_name == 'AMRI_v2':
            slopes, se, ci_lo, ci_hi, weights = method_func(X, Y, alpha)
        else:
            slopes, se, ci_lo, ci_hi = method_func(X, Y, alpha)
            weights = np.full(B, np.nan)

        valid = np.isfinite(ci_lo) & np.isfinite(ci_hi) & np.isfinite(slopes)
        if valid.sum() < B * 0.3:
            return None

        slopes_v = slopes[valid]
        ci_lo_v = ci_lo[valid]
        ci_hi_v = ci_hi[valid]
        se_v = se[valid]
        w_v = weights[valid]

        covers = (ci_lo_v <= theta_true) & (theta_true <= ci_hi_v)
        cov = covers.mean()
        widths = ci_hi_v - ci_lo_v
        biases = slopes_v - theta_true
        n_valid = int(valid.sum())

        result = {
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

        if method_name == 'AMRI_v2':
            result['avg_weight'] = round(float(np.nanmean(w_v)), 4)
            result['pct_full_robust'] = round(float((w_v >= 0.99).mean()), 4)
            result['pct_full_naive'] = round(float((w_v <= 0.01).mean()), 4)

        return result
    except Exception as e:
        log(f"    ERROR: {dgp_name}, d={delta}, n={n}, {method_name}: {e}")
        return None


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    log("=" * 70)
    log("AMRI v2 STANDALONE SIMULATION")
    log("6 DGPs x 5 severities x 6 sample sizes x 2000 reps")
    log("Running AMRI v1 and AMRI v2 head-to-head")
    log("=" * 70)

    deltas = [0.0, 0.25, 0.5, 0.75, 1.0]
    sample_sizes = [50, 100, 250, 500, 1000, 5000]
    B_REPS = 2000

    methods = {
        'AMRI': run_amri_v1,
        'AMRI_v2': run_amri_v2,
    }

    scenarios = []
    for dgp_name in DGPS:
        for delta in deltas:
            for n in sample_sizes:
                for method_name in methods:
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
                           method_name, methods[method_name], B=B_REPS)
        te = time.time()
        if res is not None:
            results.append(res)

        if (i + 1) % 20 == 0 or i == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            pct = (i + 1) / total * 100
            log(f"  [{pct:5.1f}%] {i+1}/{total} | "
                f"{dgp_name[-15:]}, d={delta}, n={n:>5}, {method_name:<10} | "
                f"{te-ts:.1f}s | elapsed={elapsed/60:.1f}m | ETA={eta/60:.1f}m")

    df = pd.DataFrame(results)
    outpath = str(Path(__file__).resolve().parent.parent / "results" / "results_amri_v2.csv")
    df.to_csv(outpath, index=False)

    total_time = time.time() - t0
    log(f"\n{'='*70}")
    log(f"SIMULATION COMPLETE")
    log(f"{'='*70}")
    log(f"Scenarios completed: {len(df)}/{total}")
    log(f"Total time: {total_time/60:.1f} min ({total_time/3600:.2f} hours)")
    log(f"Saved: {outpath}")

    # ===== HEAD-TO-HEAD COMPARISON =====
    log(f"\n{'='*70}")
    log("HEAD-TO-HEAD: AMRI v1 vs AMRI v2")
    log(f"{'='*70}")

    v1 = df[df['method'] == 'AMRI']
    v2 = df[df['method'] == 'AMRI_v2']

    log(f"\nOverall coverage:")
    log(f"  AMRI v1: {v1['coverage'].mean():.4f} (std={v1['coverage'].std():.4f})")
    log(f"  AMRI v2: {v2['coverage'].mean():.4f} (std={v2['coverage'].std():.4f})")

    log(f"\nCoverage at delta=0 (correct model):")
    v1_d0 = v1[v1['delta'] == 0.0]['coverage']
    v2_d0 = v2[v2['delta'] == 0.0]['coverage']
    log(f"  AMRI v1: {v1_d0.mean():.4f}")
    log(f"  AMRI v2: {v2_d0.mean():.4f}")

    log(f"\nCoverage at delta=1.0 (severe misspecification):")
    v1_d1 = v1[v1['delta'] == 1.0]['coverage']
    v2_d1 = v2[v2['delta'] == 1.0]['coverage']
    log(f"  AMRI v1: {v1_d1.mean():.4f}")
    log(f"  AMRI v2: {v2_d1.mean():.4f}")

    log(f"\nAverage width:")
    log(f"  AMRI v1: {v1['avg_width'].mean():.4f}")
    log(f"  AMRI v2: {v2['avg_width'].mean():.4f}")

    log(f"\nMin coverage (worst case):")
    log(f"  AMRI v1: {v1['coverage'].min():.4f}")
    log(f"  AMRI v2: {v2['coverage'].min():.4f}")

    log(f"\nCoverage std (uniformity -- lower is better):")
    log(f"  AMRI v1: {v1['coverage'].std():.4f}")
    log(f"  AMRI v2: {v2['coverage'].std():.4f}")

    # Scenario-by-scenario comparison
    merged = v1[['dgp', 'delta', 'n', 'coverage', 'avg_width']].merge(
        v2[['dgp', 'delta', 'n', 'coverage', 'avg_width']],
        on=['dgp', 'delta', 'n'], suffixes=('_v1', '_v2'))

    v2_better_cov = (merged['coverage_v2'] > merged['coverage_v1']).sum()
    v1_better_cov = (merged['coverage_v1'] > merged['coverage_v2']).sum()
    ties = (merged['coverage_v1'] == merged['coverage_v2']).sum()

    log(f"\nScenario-by-scenario coverage wins:")
    log(f"  v2 wins: {v2_better_cov}")
    log(f"  v1 wins: {v1_better_cov}")
    log(f"  Ties:    {ties}")

    v2_narrower = (merged['avg_width_v2'] < merged['avg_width_v1']).sum()
    v1_narrower = (merged['avg_width_v1'] < merged['avg_width_v2']).sum()

    log(f"\nScenario-by-scenario width (narrower is better):")
    log(f"  v2 narrower: {v2_narrower}")
    log(f"  v1 narrower: {v1_narrower}")

    # V2-specific diagnostics
    if 'avg_weight' in v2.columns:
        log(f"\n{'='*70}")
        log("AMRI v2 BLENDING WEIGHT DIAGNOSTICS")
        log(f"{'='*70}")
        for delta in deltas:
            sub = v2[v2['delta'] == delta]
            log(f"\n  delta={delta}:")
            log(f"    avg_weight     = {sub['avg_weight'].mean():.3f}")
            log(f"    pct_full_naive = {sub['pct_full_naive'].mean():.1%}")
            log(f"    pct_full_robust= {sub['pct_full_robust'].mean():.1%}")

    # Per-DGP breakdown
    log(f"\n{'='*70}")
    log("COVERAGE BY DGP AND METHOD")
    log(f"{'='*70}")
    for dgp in sorted(df['dgp'].unique()):
        log(f"\n  {dgp}:")
        for method in ['AMRI', 'AMRI_v2']:
            sub = df[(df['dgp'] == dgp) & (df['method'] == method)]
            log(f"    {method:<10}: mean={sub['coverage'].mean():.4f}, "
                f"min={sub['coverage'].min():.4f}, "
                f"std={sub['coverage'].std():.4f}, "
                f"width={sub['avg_width'].mean():.4f}")

    log("\nDONE.")
