"""
Generalized AMRI v2 Simulation
================================

Tests AMRI v2 across multiple estimator classes:
  - Multiple Linear Regression (p=3)
  - Logistic Regression
  - Poisson Regression

For each: correct specification + 2 misspecification types.
9 DGPs x 5 delta-levels x 3 sample sizes x 3 methods x 1000 reps.

Methods compared:
  1. Naive (model-based SE only)
  2. Robust (sandwich SE only)
  3. AMRI v2 General (adaptive blending)
"""

import numpy as np
from scipy import stats
import statsmodels.api as sm
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

# Import from amri_generalized
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from amri_generalized import (
    OLSEstimator, MultipleOLSEstimator, LogisticEstimator, PoissonEstimator,
    amri_v2_general, naive_inference, robust_inference
)


# =============================================================================
# DGP GENERATORS
# =============================================================================

# --- Multiple Linear Regression DGPs ---

def dgp_mlr_correct(rng, n, delta=0.0):
    """Multiple OLS, correct specification (p=3)."""
    X = rng.standard_normal((n, 3))
    eps = rng.standard_normal(n)
    Y = 1.0 + 0.5 * X[:, 0] + 0.3 * X[:, 1] - 0.2 * X[:, 2] + eps
    return X, Y, 0.5  # true beta_1

def dgp_mlr_heteroscedastic(rng, n, delta=0.5):
    """Multiple OLS with heteroscedasticity: Var(eps) = exp(delta * X1)."""
    X = rng.standard_normal((n, 3))
    sd = np.exp(delta * X[:, 0] / 2.0)
    eps = rng.standard_normal(n) * sd
    Y = 1.0 + 0.5 * X[:, 0] + 0.3 * X[:, 1] - 0.2 * X[:, 2] + eps
    return X, Y, 0.5

def dgp_mlr_omitted(rng, n, delta=0.5):
    """Multiple OLS with omitted variable: true model has X4 correlated with X1."""
    X = rng.standard_normal((n, 3))
    # X4 is correlated with X1 (rho = 0.5)
    X4 = 0.5 * X[:, 0] + np.sqrt(1 - 0.25) * rng.standard_normal(n)
    eps = rng.standard_normal(n)
    Y = 1.0 + 0.5 * X[:, 0] + 0.3 * X[:, 1] - 0.2 * X[:, 2] + delta * X4 + eps
    # Fitting model omits X4; pseudo-true beta_1 = 0.5 + delta*0.5*Var(X4)/Var(X1)
    # Since Cov(X1,X4)=0.5 and Var(X1)=1: pseudo-true = 0.5 + delta*0.5
    true_beta1 = 0.5 + delta * 0.5
    return X, Y, true_beta1


# --- Logistic Regression DGPs ---

def dgp_logistic_correct(rng, n, delta=0.0):
    """Logistic regression, correct specification."""
    X = rng.standard_normal(n)
    logit_p = -0.5 + 1.0 * X
    prob = 1.0 / (1.0 + np.exp(-logit_p))
    Y = rng.binomial(1, prob).astype(float)
    return X, Y, 1.0  # true beta = 1.0

def dgp_logistic_link_misspec(rng, n, delta=0.5):
    """Link misspecification: true model is probit, fitted as logit.
    delta controls mixture: (1-delta)*logit + delta*probit."""
    X = rng.standard_normal(n)
    # Probit: P(Y=1) = Phi(linear_pred)
    linear_pred = -0.5 + 1.0 * X
    prob_logit = 1.0 / (1.0 + np.exp(-linear_pred))
    prob_probit = stats.norm.cdf(linear_pred)
    prob = (1 - delta) * prob_logit + delta * prob_probit
    prob = np.clip(prob, 0.001, 0.999)
    Y = rng.binomial(1, prob).astype(float)
    return X, Y, 1.0  # approximate true beta

def dgp_logistic_heterogeneity(rng, n, delta=0.5):
    """Unobserved heterogeneity: random coefficient model.
    True: beta_i ~ N(1.0, delta^2), fitted with fixed beta."""
    X = rng.standard_normal(n)
    beta_i = 1.0 + delta * rng.standard_normal(n)
    logit_p = -0.5 + beta_i * X
    prob = 1.0 / (1.0 + np.exp(-logit_p))
    prob = np.clip(prob, 0.001, 0.999)
    Y = rng.binomial(1, prob).astype(float)
    return X, Y, 1.0  # target: average beta


# --- Poisson Regression DGPs ---

def dgp_poisson_correct(rng, n, delta=0.0):
    """Poisson regression, correct specification."""
    X = rng.standard_normal(n)
    log_mu = 0.5 + 0.3 * X
    mu = np.exp(log_mu)
    Y = rng.poisson(mu).astype(float)
    return X, Y, 0.3  # true beta

def dgp_poisson_overdispersed(rng, n, delta=0.5):
    """Overdispersion: true is NegBin, fitted as Poisson.
    delta controls overdispersion (higher = more)."""
    X = rng.standard_normal(n)
    log_mu = 0.5 + 0.3 * X
    mu = np.exp(log_mu)
    # NegBin with size parameter controlling overdispersion
    # Var(Y) = mu + mu^2/size; smaller size = more overdispersion
    size = max(1.0 / max(delta, 0.01), 0.5)
    p_nb = size / (size + mu)
    Y = rng.negative_binomial(size, p_nb).astype(float)
    return X, Y, 0.3

def dgp_poisson_zero_inflated(rng, n, delta=0.3):
    """Zero inflation: extra zeros beyond Poisson.
    P(Y=0 | extra) = delta; P(Y ~ Poisson | not extra) = 1-delta."""
    X = rng.standard_normal(n)
    log_mu = 0.5 + 0.3 * X
    mu = np.exp(log_mu)
    Y_pois = rng.poisson(mu).astype(float)
    # Zero-inflate
    zero_mask = rng.random(n) < delta
    Y = np.where(zero_mask, 0.0, Y_pois)
    return X, Y, 0.3  # approximate target


# =============================================================================
# SIMULATION ENGINE
# =============================================================================

def run_simulation():
    """Run the full generalized simulation."""

    # Configuration
    REPS = 1000
    N_VALUES = [100, 300, 1000]
    DELTA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]

    # DGP registry: (name, generator_fn, estimator_class, param_index, category)
    DGPS = {
        'MLR_correct':       (dgp_mlr_correct, MultipleOLSEstimator, 1, 'MLR'),
        'MLR_hetero':        (dgp_mlr_heteroscedastic, MultipleOLSEstimator, 1, 'MLR'),
        'MLR_omitted':       (dgp_mlr_omitted, MultipleOLSEstimator, 1, 'MLR'),
        'Logistic_correct':  (dgp_logistic_correct, LogisticEstimator, 1, 'Logistic'),
        'Logistic_link':     (dgp_logistic_link_misspec, LogisticEstimator, 1, 'Logistic'),
        'Logistic_hetero':   (dgp_logistic_heterogeneity, LogisticEstimator, 1, 'Logistic'),
        'Poisson_correct':   (dgp_poisson_correct, PoissonEstimator, 1, 'Poisson'),
        'Poisson_overdispersed': (dgp_poisson_overdispersed, PoissonEstimator, 1, 'Poisson'),
        'Poisson_zero_infl': (dgp_poisson_zero_inflated, PoissonEstimator, 1, 'Poisson'),
    }

    total_scenarios = len(DGPS) * len(DELTA_VALUES) * len(N_VALUES)
    print(f"Generalized AMRI v2 Simulation")
    print(f"  DGPs: {len(DGPS)}, Deltas: {len(DELTA_VALUES)}, "
          f"Sample sizes: {N_VALUES}")
    print(f"  Methods: 3 (Naive, Robust, AMRI v2)")
    print(f"  Reps: {REPS}")
    print(f"  Total scenarios: {total_scenarios}")
    print(f"  Total fits: {total_scenarios * REPS * 3:,}")
    print()

    rng = np.random.default_rng(2026)
    results = []
    t0 = time.time()

    scenario_num = 0
    for dgp_name, (dgp_fn, est_class, param_idx, category) in DGPS.items():
        for n in N_VALUES:
            for delta in DELTA_VALUES:
                scenario_num += 1
                estimator = est_class()

                cov_naive = []
                cov_robust = []
                cov_amri = []
                width_naive = []
                width_robust = []
                width_amri = []
                weights = []
                valid = 0

                for rep in range(REPS):
                    X, Y, true_beta = dgp_fn(rng, n, delta)

                    # Naive
                    _, se_n, lo_n, hi_n = naive_inference(
                        estimator, X, Y, param_idx)
                    # Robust
                    _, se_r, lo_r, hi_r = robust_inference(
                        estimator, X, Y, param_idx)
                    # AMRI v2
                    res = amri_v2_general(
                        estimator, X, Y, param_idx)

                    if np.isnan(res['theta']):
                        continue
                    valid += 1

                    cov_naive.append(
                        1 if (lo_n <= true_beta <= hi_n) else 0)
                    cov_robust.append(
                        1 if (lo_r <= true_beta <= hi_r) else 0)
                    cov_amri.append(
                        1 if (res['ci_lo'] <= true_beta <= res['ci_hi']) else 0)

                    width_naive.append(hi_n - lo_n)
                    width_robust.append(hi_r - lo_r)
                    width_amri.append(res['ci_hi'] - res['ci_lo'])
                    weights.append(res['w'])

                if valid < 100:
                    print(f"  [{scenario_num}/{total_scenarios}] "
                          f"{dgp_name} n={n} d={delta}: "
                          f"TOO FEW VALID ({valid}/{REPS}), SKIPPING")
                    continue

                row = {
                    'dgp': dgp_name,
                    'category': category,
                    'n': n,
                    'delta': delta,
                    'valid_reps': valid,
                    'cov_naive': np.mean(cov_naive),
                    'cov_robust': np.mean(cov_robust),
                    'cov_amri': np.mean(cov_amri),
                    'width_naive': np.mean(width_naive),
                    'width_robust': np.mean(width_robust),
                    'width_amri': np.mean(width_amri),
                    'avg_weight': np.mean(weights),
                    'pct_full_robust': np.mean([w >= 0.99 for w in weights]),
                    'pct_full_naive': np.mean([w <= 0.01 for w in weights]),
                }
                results.append(row)

                elapsed = time.time() - t0
                eta = elapsed / scenario_num * (total_scenarios - scenario_num)
                print(f"  [{scenario_num}/{total_scenarios}] "
                      f"{dgp_name:25s} n={n:4d} d={delta:.2f}: "
                      f"Naive={row['cov_naive']:.3f} "
                      f"Robust={row['cov_robust']:.3f} "
                      f"AMRI={row['cov_amri']:.3f} "
                      f"w={row['avg_weight']:.3f} "
                      f"({elapsed:.0f}s, ETA {eta:.0f}s)")

    df = pd.DataFrame(results)

    # Save results
    outpath = os.path.join(os.path.dirname(__file__), '..', 'results',
                           'results_generalized.csv')
    df.to_csv(outpath, index=False)
    print(f"\nResults saved to {outpath}")

    # Summary
    print()
    print("=" * 72)
    print("SUMMARY BY ESTIMATOR CLASS")
    print("=" * 72)

    for category in ['MLR', 'Logistic', 'Poisson']:
        sub = df[df['category'] == category]
        if len(sub) == 0:
            continue
        print(f"\n  {category}:")
        print(f"    {'Metric':>20s}  {'Naive':>8s}  {'Robust':>8s}  {'AMRI v2':>8s}")
        print(f"    {'Mean coverage':>20s}  "
              f"{sub['cov_naive'].mean():>8.4f}  "
              f"{sub['cov_robust'].mean():>8.4f}  "
              f"{sub['cov_amri'].mean():>8.4f}")
        print(f"    {'Min coverage':>20s}  "
              f"{sub['cov_naive'].min():>8.4f}  "
              f"{sub['cov_robust'].min():>8.4f}  "
              f"{sub['cov_amri'].min():>8.4f}")
        print(f"    {'Cov std dev':>20s}  "
              f"{sub['cov_naive'].std():>8.4f}  "
              f"{sub['cov_robust'].std():>8.4f}  "
              f"{sub['cov_amri'].std():>8.4f}")
        print(f"    {'Mean width':>20s}  "
              f"{sub['width_naive'].mean():>8.4f}  "
              f"{sub['width_robust'].mean():>8.4f}  "
              f"{sub['width_amri'].mean():>8.4f}")
        print(f"    {'Avg weight':>20s}  "
              f"{'---':>8s}  "
              f"{'---':>8s}  "
              f"{sub['avg_weight'].mean():>8.4f}")

        # Per-DGP breakdown
        for dgp_name in sub['dgp'].unique():
            dsub = sub[sub['dgp'] == dgp_name]
            print(f"\n    {dgp_name}:")
            print(f"      {'delta':>8s}  {'n':>5s}  "
                  f"{'Naive':>7s}  {'Robust':>7s}  {'AMRI':>7s}  "
                  f"{'w':>6s}")
            for _, r in dsub.iterrows():
                print(f"      {r['delta']:>8.2f}  {int(r['n']):>5d}  "
                      f"{r['cov_naive']:>7.3f}  {r['cov_robust']:>7.3f}  "
                      f"{r['cov_amri']:>7.3f}  {r['avg_weight']:>6.3f}")

    # Overall comparison
    print()
    print("=" * 72)
    print("OVERALL COMPARISON (all DGPs)")
    print("=" * 72)
    print(f"  {'Metric':>25s}  {'Naive':>8s}  {'Robust':>8s}  {'AMRI v2':>8s}")
    print(f"  {'Mean coverage':>25s}  "
          f"{df['cov_naive'].mean():>8.4f}  "
          f"{df['cov_robust'].mean():>8.4f}  "
          f"{df['cov_amri'].mean():>8.4f}")
    print(f"  {'Min coverage':>25s}  "
          f"{df['cov_naive'].min():>8.4f}  "
          f"{df['cov_robust'].min():>8.4f}  "
          f"{df['cov_amri'].min():>8.4f}")
    print(f"  {'Coverage std':>25s}  "
          f"{df['cov_naive'].std():>8.4f}  "
          f"{df['cov_robust'].std():>8.4f}  "
          f"{df['cov_amri'].std():>8.4f}")
    print(f"  {'Mean width':>25s}  "
          f"{df['width_naive'].mean():>8.4f}  "
          f"{df['width_robust'].mean():>8.4f}  "
          f"{df['width_amri'].mean():>8.4f}")

    # AMRI v2 advantages
    amri_beats_naive = (df['cov_amri'] >= df['cov_naive']).sum()
    amri_beats_robust_width = (df['width_amri'] <= df['width_robust']).sum()
    print(f"\n  AMRI v2 >= Naive coverage: {amri_beats_naive}/{len(df)} scenarios "
          f"({100*amri_beats_naive/len(df):.1f}%)")
    print(f"  AMRI v2 <= Robust width:  {amri_beats_robust_width}/{len(df)} scenarios "
          f"({100*amri_beats_robust_width/len(df):.1f}%)")

    total_time = time.time() - t0
    print(f"\n  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")

    return df


if __name__ == '__main__':
    df = run_simulation()
