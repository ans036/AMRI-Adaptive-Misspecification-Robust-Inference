"""
Full Simulation Framework: Inference Under Model Misspecification
================================================================
Implements 6 DGPs × 5 severity levels × 6 sample sizes × 10 methods × B replications.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit  # logistic function
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, QuantileRegressor
from joblib import Parallel, delayed
import warnings
import time
import os
import json

warnings.filterwarnings('ignore')

# ============================================================================
# DATA GENERATING PROCESSES
# ============================================================================

def dgp_nonlinearity(n, delta, rng):
    """DGP1: Functional form misspecification (linear fit to nonlinear data)."""
    X = rng.standard_normal(n)
    epsilon = rng.standard_normal(n)
    Y = X + delta * X**2 + epsilon
    # Pseudo-true slope: Cov(Y,X)/Var(X) = 1 + delta*E[X^3]/E[X^2] = 1 (by symmetry)
    theta_true = 1.0
    return X.reshape(-1, 1), Y, theta_true


def dgp_heavy_tails(n, delta, rng):
    """DGP2: Distributional misspecification (heavy-tailed errors)."""
    X = rng.standard_normal(n)
    # Map delta to degrees of freedom: delta=0 -> normal, delta=1 -> t(2)
    if delta == 0:
        epsilon = rng.standard_normal(n)
    else:
        df = max(2.1, 10 - 8 * delta)  # df goes from 10 (mild) to 2.1 (extreme)
        epsilon = rng.standard_t(df, n)
        epsilon = epsilon / np.sqrt(df / (df - 2))  # standardize to variance 1
    Y = X + epsilon
    theta_true = 1.0
    return X.reshape(-1, 1), Y, theta_true


def dgp_heteroscedasticity(n, delta, rng):
    """DGP3: Variance misspecification (heteroscedastic errors, fit assumes homoscedastic)."""
    X = rng.standard_normal(n)
    sigma = np.exp(delta * X / 2)  # Var(eps|X) = exp(delta * X)
    epsilon = rng.standard_normal(n) * sigma
    Y = X + epsilon
    theta_true = 1.0
    return X.reshape(-1, 1), Y, theta_true


def dgp_omitted_variable(n, delta, rng):
    """DGP4: Omitted variable bias."""
    rho = 0.5
    # Generate correlated X1, X2
    Z = rng.standard_normal((n, 2))
    X1 = Z[:, 0]
    X2 = rho * Z[:, 0] + np.sqrt(1 - rho**2) * Z[:, 1]
    epsilon = rng.standard_normal(n)
    Y = X1 + delta * X2 + epsilon  # beta1=1, beta2=delta
    # Pseudo-true: alpha1* = 1 + delta * rho = 1 + 0.5*delta
    # But the TARGET is beta1 = 1 (the partial effect of X1)
    # Under OLS of Y on X1: alpha1_hat -> 1 + delta*rho
    theta_true_pseudo = 1.0 + delta * rho  # what OLS converges to
    theta_true_causal = 1.0  # what we want
    return X1.reshape(-1, 1), Y, theta_true_pseudo


def dgp_link_function(n, delta, rng):
    """DGP5: Link function misspecification (probit data, logit fit)."""
    X = rng.standard_normal(n)
    beta = 0.5 + 2 * delta  # increases divergence between probit/logit
    p_true = stats.norm.cdf(beta * X)  # Probit link
    Y = rng.binomial(1, p_true)
    # Pseudo-true logit coefficient ≈ beta * (pi/sqrt(3)) / 1 ≈ beta * 1.6
    # Actually probit_coeff ≈ logit_coeff / 1.7
    theta_true = beta / 1.7  # approximate pseudo-true logit coefficient
    return X.reshape(-1, 1), Y, theta_true


def dgp_clustering(n, delta, rng):
    """DGP6: Ignored clustering/dependence structure."""
    G = max(10, n // 10)  # number of clusters
    m = n // G  # cluster size
    actual_n = G * m
    tau = np.sqrt(delta)  # cluster random effect SD
    u = rng.standard_normal(G) * tau  # cluster effects
    X = rng.standard_normal(actual_n)
    cluster_ids = np.repeat(np.arange(G), m)
    epsilon = rng.standard_normal(actual_n)
    Y = X + u[cluster_ids] + epsilon
    theta_true = 1.0
    return X[:actual_n].reshape(-1, 1), Y[:actual_n], theta_true


DGP_FUNCTIONS = {
    'DGP1_nonlinearity': dgp_nonlinearity,
    'DGP2_heavy_tails': dgp_heavy_tails,
    'DGP3_heteroscedasticity': dgp_heteroscedasticity,
    'DGP4_omitted_variable': dgp_omitted_variable,
    'DGP5_link_function': dgp_link_function,
    'DGP6_clustering': dgp_clustering,
}

# ============================================================================
# INFERENCE METHODS
# ============================================================================

def method_naive(X, Y, alpha=0.05):
    """Method 1: Naive OLS + model-based SE."""
    Xa = sm.add_constant(X)
    try:
        model = sm.OLS(Y, Xa).fit()
        theta = model.params[1]
        ci = model.conf_int(alpha=alpha)[1]
        se = model.bse[1]
        return theta, se, ci[0], ci[1]
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def method_sandwich_hc3(X, Y, alpha=0.05):
    """Method 2: OLS + HC3 sandwich standard errors."""
    Xa = sm.add_constant(X)
    try:
        model = sm.OLS(Y, Xa).fit(cov_type='HC3')
        theta = model.params[1]
        ci = model.conf_int(alpha=alpha)[1]
        se = model.bse[1]
        return theta, se, ci[0], ci[1]
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def method_sandwich_hc0(X, Y, alpha=0.05):
    """Method 3: OLS + HC0 (White's original) sandwich SE."""
    Xa = sm.add_constant(X)
    try:
        model = sm.OLS(Y, Xa).fit(cov_type='HC0')
        theta = model.params[1]
        ci = model.conf_int(alpha=alpha)[1]
        se = model.bse[1]
        return theta, se, ci[0], ci[1]
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def method_pairs_bootstrap(X, Y, alpha=0.05, B_boot=999, _seed=None):
    """Method 4: Pairs bootstrap (percentile CI)."""
    n = len(Y)
    Xa = sm.add_constant(X)
    try:
        base_model = sm.OLS(Y, Xa).fit()
        theta = base_model.params[1]
        rng_boot = np.random.default_rng(_seed)
        boot_thetas = np.empty(B_boot)
        for b in range(B_boot):
            idx = rng_boot.integers(0, n, size=n)
            try:
                m = sm.OLS(Y[idx], Xa[idx]).fit()
                boot_thetas[b] = m.params[1]
            except Exception:
                boot_thetas[b] = np.nan
        boot_thetas = boot_thetas[~np.isnan(boot_thetas)]
        if len(boot_thetas) < 100:
            return np.nan, np.nan, np.nan, np.nan
        ci_low = np.percentile(boot_thetas, 100 * alpha / 2)
        ci_high = np.percentile(boot_thetas, 100 * (1 - alpha / 2))
        se = np.std(boot_thetas)
        return theta, se, ci_low, ci_high
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def method_wild_bootstrap(X, Y, alpha=0.05, B_boot=999, _seed=None):
    """Method 5: Wild bootstrap (Rademacher) with percentile CI."""
    n = len(Y)
    Xa = sm.add_constant(X)
    try:
        base_model = sm.OLS(Y, Xa).fit()
        theta = base_model.params[1]
        resid = base_model.resid
        y_fitted = base_model.fittedvalues
        rng_boot = np.random.default_rng(_seed)
        boot_thetas = np.empty(B_boot)
        for b in range(B_boot):
            v = rng_boot.choice([-1, 1], size=n)  # Rademacher
            Y_star = y_fitted + resid * v
            try:
                m = sm.OLS(Y_star, Xa).fit()
                boot_thetas[b] = m.params[1]
            except Exception:
                boot_thetas[b] = np.nan
        boot_thetas = boot_thetas[~np.isnan(boot_thetas)]
        if len(boot_thetas) < 100:
            return np.nan, np.nan, np.nan, np.nan
        ci_low = np.percentile(boot_thetas, 100 * alpha / 2)
        ci_high = np.percentile(boot_thetas, 100 * (1 - alpha / 2))
        se = np.std(boot_thetas)
        return theta, se, ci_low, ci_high
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def method_bootstrap_t(X, Y, alpha=0.05, B_boot=999, _seed=None):
    """Method 6: Studentized bootstrap (bootstrap-t)."""
    n = len(Y)
    Xa = sm.add_constant(X)
    try:
        base_model = sm.OLS(Y, Xa).fit(cov_type='HC3')
        theta = base_model.params[1]
        se_base = base_model.bse[1]
        rng_boot = np.random.default_rng(_seed)
        t_stars = []
        for b in range(B_boot):
            idx = rng_boot.integers(0, n, size=n)
            try:
                m = sm.OLS(Y[idx], Xa[idx]).fit(cov_type='HC3')
                t_star = (m.params[1] - theta) / m.bse[1]
                t_stars.append(t_star)
            except Exception:
                pass
        t_stars = np.array(t_stars)
        if len(t_stars) < 100:
            return np.nan, np.nan, np.nan, np.nan
        q_low = np.percentile(t_stars, 100 * alpha / 2)
        q_high = np.percentile(t_stars, 100 * (1 - alpha / 2))
        ci_low = theta - q_high * se_base
        ci_high = theta - q_low * se_base
        return theta, se_base, ci_low, ci_high
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def method_conformal_split(X, Y, alpha=0.05):
    """Method 7: Split conformal prediction interval (evaluated at X mean)."""
    n = len(Y)
    split = n // 2
    X_train, Y_train = X[:split], Y[:split]
    X_cal, Y_cal = X[split:], Y[split:]

    try:
        model = LinearRegression().fit(X_train, Y_train)
        residuals_cal = np.abs(Y_cal - model.predict(X_cal))
        q = np.quantile(residuals_cal, 1 - alpha, method='higher')

        # For comparison with parametric methods, evaluate at X = mean(X)
        # and also compute the "slope" CI via regression on calibration set
        # Actually, conformal gives prediction intervals, not parameter CIs
        # We'll report coverage of prediction intervals on calibration data
        Xa = sm.add_constant(X)
        ols = sm.OLS(Y, Xa).fit()
        theta = ols.params[1]

        # Construct a pseudo-CI for the slope using conformal residuals
        # Width of prediction interval at X=1 minus width at X=0 gives slope uncertainty
        pred_at_0 = model.predict(np.array([[0]]))[0]
        pred_at_1 = model.predict(np.array([[1]]))[0]
        slope_est = pred_at_1 - pred_at_0

        # Use conformal residual quantile as SE proxy
        se_proxy = q / np.sqrt(n)
        ci_low = theta - stats.norm.ppf(1 - alpha/2) * se_proxy
        ci_high = theta + stats.norm.ppf(1 - alpha/2) * se_proxy

        return theta, se_proxy, ci_low, ci_high
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def method_jackknife_plus(X, Y, alpha=0.05):
    """Method 8: Jackknife+ inspired CI for regression coefficient."""
    n = len(Y)
    Xa = sm.add_constant(X)
    try:
        # Full-data estimate
        full_model = sm.OLS(Y, Xa).fit()
        theta_full = full_model.params[1]

        # Leave-one-out estimates
        loo_thetas = np.empty(n)
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            try:
                m = sm.OLS(Y[mask], Xa[mask]).fit()
                loo_thetas[i] = m.params[1]
            except Exception:
                loo_thetas[i] = np.nan

        valid = ~np.isnan(loo_thetas)
        if valid.sum() < n * 0.8:
            return np.nan, np.nan, np.nan, np.nan

        loo_thetas = loo_thetas[valid]
        # Jackknife+ CI: use the distribution of LOO estimates
        residuals = loo_thetas - theta_full
        se = np.std(loo_thetas) * np.sqrt(n)  # jackknife SE
        ci_low = np.quantile(loo_thetas, alpha / 2)
        ci_high = np.quantile(loo_thetas, 1 - alpha / 2)
        # Better: use jackknife SE for CI
        se_jk = np.sqrt((n - 1) / n * np.sum((loo_thetas - np.mean(loo_thetas))**2))
        ci_low = theta_full - stats.t.ppf(1 - alpha/2, n-2) * se_jk
        ci_high = theta_full + stats.t.ppf(1 - alpha/2, n-2) * se_jk
        return theta_full, se_jk, ci_low, ci_high
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def method_bayesian_normal(X, Y, alpha=0.05):
    """Method 9: Approximate Bayesian posterior (conjugate normal-normal)."""
    n = len(Y)
    Xa = sm.add_constant(X)
    try:
        # Conjugate Bayesian linear regression with vague prior
        # Prior: beta ~ N(0, 100*I), sigma^2 ~ IG(0.01, 0.01)
        # Posterior (approximately):
        prior_precision = np.eye(Xa.shape[1]) / 100.0
        XtX = Xa.T @ Xa
        XtY = Xa.T @ Y
        post_precision = XtX + prior_precision
        post_var = np.linalg.inv(post_precision)
        post_mean = post_var @ (XtY)

        # Residual variance estimate
        resid = Y - Xa @ post_mean
        s2 = np.sum(resid**2) / (n - 2)

        theta = post_mean[1]
        # Model-based posterior SD (this is what breaks under misspecification)
        se = np.sqrt(s2 * post_var[1, 1])
        ci_low = theta - stats.t.ppf(1 - alpha/2, n-2) * se
        ci_high = theta + stats.t.ppf(1 - alpha/2, n-2) * se
        return theta, se, ci_low, ci_high
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def method_amri(X, Y, alpha=0.05):
    """Method 10: Adaptive Misspecification-Robust Inference (AMRI v1) - hard switching."""
    n = len(Y)
    Xa = sm.add_constant(X)
    try:
        # Step 1: Get both naive and sandwich estimates
        model_naive = sm.OLS(Y, Xa).fit()
        model_sandwich = sm.OLS(Y, Xa).fit(cov_type='HC3')

        theta = model_naive.params[1]
        se_naive = model_naive.bse[1]
        se_sandwich = model_sandwich.bse[1]

        # Step 2: Misspecification diagnostic
        # Ratio of sandwich to model-based SE
        se_ratio = se_sandwich / se_naive if se_naive > 0 else 999

        # Adaptive threshold (gets tighter with more data)
        threshold = 1 + 2 / np.sqrt(n)

        if se_ratio > threshold or se_ratio < 1 / threshold:
            # Misspecification detected -> use sandwich SE with inflation
            se_used = se_sandwich * 1.05  # small inflation for safety
            ci_low = theta - stats.t.ppf(1 - alpha/2, n-2) * se_used
            ci_high = theta + stats.t.ppf(1 - alpha/2, n-2) * se_used
        else:
            # No misspecification detected -> use model-based (more efficient)
            se_used = se_naive
            ci_low = model_naive.conf_int(alpha=alpha)[1][0]
            ci_high = model_naive.conf_int(alpha=alpha)[1][1]

        return theta, se_used, ci_low, ci_high
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def method_amri_v2(X, Y, alpha=0.05, c1=1.0, c2=2.0):
    """Method 11: AMRI v2 - Soft-thresholding variant (no discontinuity)."""
    n = len(Y)
    Xa = sm.add_constant(X)
    try:
        model_naive = sm.OLS(Y, Xa).fit()
        model_sandwich = sm.OLS(Y, Xa).fit(cov_type='HC3')

        theta = model_naive.params[1]
        se_naive = model_naive.bse[1]
        se_sandwich = model_sandwich.bse[1]

        # Soft-thresholding blending weight
        ratio = se_sandwich / max(se_naive, 1e-10)
        log_ratio = abs(np.log(ratio))
        lower = c1 / np.sqrt(n)
        upper = c2 / np.sqrt(n)

        if upper <= lower:
            w = 1.0 if log_ratio > lower else 0.0
        else:
            w = np.clip((log_ratio - lower) / (upper - lower), 0.0, 1.0)

        se_used = (1 - w) * se_naive + w * se_sandwich
        t_val = stats.t.ppf(1 - alpha/2, n - 2)
        ci_low = theta - t_val * se_used
        ci_high = theta + t_val * se_used

        return theta, se_used, ci_low, ci_high
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


# Methods for GLM (DGP5)
def method_naive_glm(X, Y, alpha=0.05):
    """Naive logistic regression + model-based SE."""
    Xa = sm.add_constant(X)
    try:
        model = sm.Logit(Y, Xa).fit(disp=0)
        theta = model.params[1]
        ci = model.conf_int(alpha=alpha)[1]
        se = model.bse[1]
        return theta, se, ci[0], ci[1]
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def method_sandwich_glm(X, Y, alpha=0.05):
    """Logistic regression + sandwich SE."""
    Xa = sm.add_constant(X)
    try:
        model = sm.Logit(Y, Xa).fit(disp=0, cov_type='HC0')
        theta = model.params[1]
        ci = model.conf_int(alpha=alpha)[1]
        se = model.bse[1]
        return theta, se, ci[0], ci[1]
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


# Map method names to functions
METHODS_LINEAR = {
    'Naive_OLS': method_naive,
    'Sandwich_HC3': method_sandwich_hc3,
    'Sandwich_HC0': method_sandwich_hc0,
    'Pairs_Bootstrap': method_pairs_bootstrap,
    'Wild_Bootstrap': method_wild_bootstrap,
    'Bootstrap_t': method_bootstrap_t,
    'Split_Conformal': method_conformal_split,
    'Jackknife': method_jackknife_plus,
    'Bayesian_Normal': method_bayesian_normal,
    'AMRI': method_amri,
    'AMRI_v2': method_amri_v2,
}

METHODS_GLM = {
    'Naive_GLM': method_naive_glm,
    'Sandwich_GLM': method_sandwich_glm,
}


# ============================================================================
# SIMULATION RUNNER
# ============================================================================

def run_single_rep(dgp_name, dgp_func, delta, n, method_name, method_func, seed, alpha=0.05):
    """Run a single replication: generate data, apply method, return result."""
    rng = np.random.default_rng(seed)
    X, Y, theta_true = dgp_func(n, delta, rng)
    # Pass a unique seed to bootstrap methods so each rep has independent bootstrap samples
    bootstrap_methods = ('Pairs_Bootstrap', 'Wild_Bootstrap', 'Bootstrap_t')
    if method_name in bootstrap_methods:
        boot_seed = np.random.SeedSequence(seed).spawn(1)[0].generate_state(1)[0]
        theta_hat, se_hat, ci_low, ci_high = method_func(X.ravel() if X.ndim > 1 else X, Y, alpha, _seed=boot_seed)
    else:
        theta_hat, se_hat, ci_low, ci_high = method_func(X.ravel() if X.ndim > 1 else X, Y, alpha)
    covers = 1 if (ci_low <= theta_true <= ci_high) else 0
    width = ci_high - ci_low
    bias = theta_hat - theta_true
    return {
        'theta_hat': theta_hat,
        'se_hat': se_hat,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'covers': covers,
        'width': width,
        'bias': bias,
        'theta_true': theta_true,
    }


def run_scenario(dgp_name, delta, n, method_name, B=1000, alpha=0.05, master_seed=20260228):
    """Run B replications of a single scenario."""
    dgp_func = DGP_FUNCTIONS[dgp_name]

    # Choose appropriate method set
    if dgp_name == 'DGP5_link_function':
        if method_name in METHODS_GLM:
            method_func = METHODS_GLM[method_name]
        else:
            return None  # Skip linear methods for GLM DGP
    else:
        if method_name in METHODS_LINEAR:
            method_func = METHODS_LINEAR[method_name]
        else:
            return None

    rng_master = np.random.SeedSequence(master_seed)
    child_seeds = rng_master.spawn(B)

    results = []
    for b in range(B):
        seed = child_seeds[b].generate_state(1)[0]
        res = run_single_rep(dgp_name, dgp_func, delta, n, method_name, method_func, seed, alpha)
        results.append(res)

    df = pd.DataFrame(results)
    valid = df.dropna()

    if len(valid) < B * 0.5:
        return None

    summary = {
        'dgp': dgp_name,
        'delta': delta,
        'n': n,
        'method': method_name,
        'B': len(valid),
        'coverage': valid['covers'].mean(),
        'coverage_mc_se': np.sqrt(valid['covers'].mean() * (1 - valid['covers'].mean()) / len(valid)),
        'avg_width': valid['width'].mean(),
        'median_width': valid['width'].median(),
        'bias': valid['bias'].mean(),
        'rmse': np.sqrt((valid['bias']**2).mean()),
        'se_mean': valid['se_hat'].mean(),
        'theta_true': valid['theta_true'].iloc[0],
    }
    return summary


def run_full_simulation(dgps, deltas, sample_sizes, methods, B=1000, n_jobs=1):
    """Run the full simulation across all scenarios."""
    scenarios = []
    for dgp in dgps:
        method_set = METHODS_GLM if dgp == 'DGP5_link_function' else METHODS_LINEAR
        for delta in deltas:
            for n in sample_sizes:
                for method_name in methods:
                    if method_name in method_set:
                        scenarios.append((dgp, delta, n, method_name))

    print(f"Total scenarios: {len(scenarios)}")
    print(f"Total simulation runs: {len(scenarios) * B:,}")
    print(f"Starting simulation...")

    all_results = []
    total = len(scenarios)
    for i, (dgp, delta, n, method_name) in enumerate(scenarios):
        t0 = time.time()
        result = run_scenario(dgp, delta, n, method_name, B=B)
        elapsed = time.time() - t0
        if result is not None:
            all_results.append(result)
        if (i + 1) % 10 == 0 or i == 0:
            pct = (i + 1) / total * 100
            print(f"  [{pct:5.1f}%] Scenario {i+1}/{total}: {dgp}, delta={delta}, n={n}, {method_name} ({elapsed:.1f}s)")

    return pd.DataFrame(all_results)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    import sys

    from pathlib import Path
    mode = sys.argv[1] if len(sys.argv) > 1 else 'pilot'
    output_dir = str(Path(__file__).resolve().parent.parent / "results")

    if mode == 'pilot':
        print("=" * 70)
        print("PILOT STUDY: Quick validation run")
        print("=" * 70)
        dgps = ['DGP1_nonlinearity', 'DGP3_heteroscedasticity']
        deltas = [0.0, 0.5, 1.0]
        sample_sizes = [100, 500]
        methods = ['Naive_OLS', 'Sandwich_HC3', 'Pairs_Bootstrap', 'Bayesian_Normal', 'AMRI']
        B = 500

    elif mode == 'full':
        print("=" * 70)
        print("FULL SIMULATION")
        print("=" * 70)
        dgps = ['DGP1_nonlinearity', 'DGP2_heavy_tails', 'DGP3_heteroscedasticity',
                'DGP4_omitted_variable', 'DGP6_clustering']
        deltas = [0.0, 0.25, 0.5, 0.75, 1.0]
        sample_sizes = [50, 100, 250, 500, 1000, 5000]
        methods = list(METHODS_LINEAR.keys())
        B = 2000

    elif mode == 'glm':
        print("=" * 70)
        print("GLM SIMULATION (DGP5: Link function misspecification)")
        print("=" * 70)
        dgps = ['DGP5_link_function']
        deltas = [0.0, 0.25, 0.5, 0.75, 1.0]
        sample_sizes = [100, 250, 500, 1000, 5000]
        methods = list(METHODS_GLM.keys())
        B = 2000

    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)

    t_start = time.time()
    results_df = run_full_simulation(dgps, deltas, sample_sizes, methods, B=B)
    t_total = time.time() - t_start

    # Save results
    outfile = os.path.join(output_dir, f"results_{mode}.csv")
    results_df.to_csv(outfile, index=False)
    print(f"\nResults saved to: {outfile}")
    print(f"Total time: {t_total:.1f}s ({t_total/60:.1f} min)")
    print(f"Total scenarios completed: {len(results_df)}")

    # Quick summary
    print("\n" + "=" * 70)
    print("QUICK SUMMARY: Coverage by Method (averaged across all scenarios)")
    print("=" * 70)
    summary = results_df.groupby('method')['coverage'].agg(['mean', 'std', 'min', 'max'])
    print(summary.to_string())
