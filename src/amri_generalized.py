"""
Generalized AMRI v2: Adaptive Misspecification-Robust Inference
================================================================

Extends AMRI v2 from simple linear regression to general estimation:
  - Multiple linear regression (p > 1)
  - Logistic regression (GLM with Binomial/Logit)
  - Poisson regression (GLM with Poisson/Log)
  - Any estimator providing model-based and robust standard errors

Architecture:
  AMRIEstimator (ABC)        -- abstract base
    OLSEstimator             -- simple OLS (validates against existing code)
    MultipleOLSEstimator     -- multiple regression
    LogisticEstimator        -- logistic via statsmodels GLM
    PoissonEstimator         -- Poisson via statsmodels GLM

  amri_v2_general()          -- THE core function, estimator-agnostic

The adaptive logic (ratio -> log-ratio -> blending weight -> SE blend) is
IDENTICAL across all estimator classes. Only the fitting and SE computation
differ.

Reference: threshold_proof.py for formal justification of c1, c2.
"""

import numpy as np
from scipy import stats
from abc import ABC, abstractmethod
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# ABSTRACT BASE CLASS
# =============================================================================

class AMRIEstimator(ABC):
    """Abstract base for any estimator compatible with AMRI v2.

    Subclasses must provide:
        fit(X, Y)        -> dict with 'theta', 'se_model', 'se_robust', 'dof', etc.
        name             -> string identifier
    """

    @abstractmethod
    def fit(self, X, Y):
        """Fit the model and return both model-based and robust SEs.

        Returns dict with keys:
            'theta': array of parameter estimates
            'se_model': array of model-based SEs
            'se_robust': array of robust/sandwich SEs
            'dof': degrees of freedom for t-distribution (0 = use z)
            'converged': bool
        """
        pass

    @property
    @abstractmethod
    def name(self):
        pass


# =============================================================================
# CORE AMRI v2 FUNCTION (Estimator-Agnostic)
# =============================================================================

def amri_v2_general(estimator, X, Y, param_index=1, alpha=0.05,
                    c1=1.0, c2=2.0):
    """Generalized AMRI v2: works with ANY estimator that provides
    model-based SE and robust SE.

    Parameters
    ----------
    estimator : AMRIEstimator
        Fitted estimator class providing se_model and se_robust.
    X : array (n, p) or (n,)
        Predictor matrix (without intercept; intercept is added internally).
    Y : array (n,)
        Response vector.
    param_index : int
        Which parameter to do inference on (0=intercept, 1=first predictor, etc.)
    alpha : float
        Significance level.
    c1, c2 : float
        Blending thresholds (justified in threshold_proof.py).

    Returns
    -------
    dict with keys: theta, se, ci_lo, ci_hi, w, se_model, se_robust, ratio, dof
    """
    n = len(Y)

    # Step 1: Fit and get both SEs
    fit_result = estimator.fit(X, Y)

    if not fit_result['converged']:
        return {
            'theta': np.nan, 'se': np.nan,
            'ci_lo': np.nan, 'ci_hi': np.nan,
            'w': np.nan, 'se_model': np.nan, 'se_robust': np.nan,
            'ratio': np.nan, 'dof': 0
        }

    theta_hat = fit_result['theta'][param_index]
    se_model = fit_result['se_model'][param_index]
    se_robust = fit_result['se_robust'][param_index]
    dof = fit_result['dof']

    # Step 2-5: AMRI v2 adaptive logic (GENERIC)
    ratio = se_robust / max(se_model, 1e-10)
    s = abs(np.log(ratio))
    lo = c1 / np.sqrt(n)
    hi = c2 / np.sqrt(n)

    if hi <= lo:
        w = 1.0 if s > lo else 0.0
    else:
        w = float(np.clip((s - lo) / (hi - lo), 0.0, 1.0))

    se_amri = (1 - w) * se_model + w * se_robust

    # Step 6: CI using appropriate critical value
    if dof > 0:
        t_val = stats.t.ppf(1 - alpha / 2, dof)
    else:
        t_val = stats.norm.ppf(1 - alpha / 2)

    ci_lo = theta_hat - t_val * se_amri
    ci_hi = theta_hat + t_val * se_amri

    return {
        'theta': theta_hat, 'se': se_amri,
        'ci_lo': ci_lo, 'ci_hi': ci_hi,
        'w': w, 'se_model': se_model, 'se_robust': se_robust,
        'ratio': ratio, 'dof': dof
    }


# =============================================================================
# ESTIMATOR 1: Simple OLS (validates against existing AMRI v2)
# =============================================================================

class OLSEstimator(AMRIEstimator):
    """Simple linear regression using statsmodels OLS."""

    @property
    def name(self):
        return "OLS"

    def fit(self, X, Y):
        try:
            if X.ndim == 1:
                Xa = sm.add_constant(X)
            else:
                Xa = sm.add_constant(X)

            model_naive = sm.OLS(Y, Xa).fit()
            model_robust = sm.OLS(Y, Xa).fit(cov_type='HC3')

            return {
                'theta': model_naive.params,
                'se_model': model_naive.bse,
                'se_robust': model_robust.bse,
                'dof': int(model_naive.df_resid),
                'converged': True
            }
        except Exception:
            p = 2 if X.ndim == 1 else X.shape[1] + 1
            return {
                'theta': np.full(p, np.nan),
                'se_model': np.full(p, np.nan),
                'se_robust': np.full(p, np.nan),
                'dof': 0, 'converged': False
            }


# =============================================================================
# ESTIMATOR 2: Multiple OLS (p > 1 predictors)
# =============================================================================

class MultipleOLSEstimator(AMRIEstimator):
    """Multiple linear regression with p >= 1 predictors."""

    @property
    def name(self):
        return "MultipleOLS"

    def fit(self, X, Y):
        try:
            Xa = sm.add_constant(X)
            model_naive = sm.OLS(Y, Xa).fit()
            model_robust = sm.OLS(Y, Xa).fit(cov_type='HC3')

            return {
                'theta': model_naive.params,
                'se_model': model_naive.bse,
                'se_robust': model_robust.bse,
                'dof': int(model_naive.df_resid),
                'converged': True
            }
        except Exception:
            p = X.shape[1] + 1
            return {
                'theta': np.full(p, np.nan),
                'se_model': np.full(p, np.nan),
                'se_robust': np.full(p, np.nan),
                'dof': 0, 'converged': False
            }


# =============================================================================
# ESTIMATOR 3: Logistic Regression (GLM Binomial/Logit)
# =============================================================================

class LogisticEstimator(AMRIEstimator):
    """Logistic regression via statsmodels GLM.

    Model-based SE: from Fisher information (inverse Hessian)
    Robust SE: HC1 sandwich (no HC3 analog for GLMs in statsmodels)
    Critical value: z (normal), since GLM uses asymptotic normality
    """

    @property
    def name(self):
        return "Logistic"

    def fit(self, X, Y):
        try:
            Xa = sm.add_constant(X)
            family = sm.families.Binomial()

            model_naive = sm.GLM(Y, Xa, family=family).fit()
            model_robust = sm.GLM(Y, Xa, family=family).fit(
                cov_type='HC1'
            )

            return {
                'theta': model_naive.params,
                'se_model': model_naive.bse,
                'se_robust': model_robust.bse,
                'dof': 0,  # use z for GLMs
                'converged': True
            }
        except Exception:
            p = X.shape[1] + 1 if X.ndim > 1 else 2
            return {
                'theta': np.full(p, np.nan),
                'se_model': np.full(p, np.nan),
                'se_robust': np.full(p, np.nan),
                'dof': 0, 'converged': False
            }


# =============================================================================
# ESTIMATOR 4: Poisson Regression (GLM Poisson/Log)
# =============================================================================

class PoissonEstimator(AMRIEstimator):
    """Poisson regression via statsmodels GLM.

    Model-based SE: from Fisher information (assumes Var(Y|X) = mu)
    Robust SE: HC1 sandwich (valid under overdispersion)
    Critical value: z (normal)
    """

    @property
    def name(self):
        return "Poisson"

    def fit(self, X, Y):
        try:
            Xa = sm.add_constant(X)
            family = sm.families.Poisson()

            model_naive = sm.GLM(Y, Xa, family=family).fit()
            model_robust = sm.GLM(Y, Xa, family=family).fit(
                cov_type='HC1'
            )

            return {
                'theta': model_naive.params,
                'se_model': model_naive.bse,
                'se_robust': model_robust.bse,
                'dof': 0,  # use z for GLMs
                'converged': True
            }
        except Exception:
            p = X.shape[1] + 1 if X.ndim > 1 else 2
            return {
                'theta': np.full(p, np.nan),
                'se_model': np.full(p, np.nan),
                'se_robust': np.full(p, np.nan),
                'dof': 0, 'converged': False
            }


# =============================================================================
# CONVENIENCE FUNCTIONS: Naive and Robust-only for comparison
# =============================================================================

def naive_inference(estimator, X, Y, param_index=1, alpha=0.05):
    """Inference using model-based SE only (no adaptation)."""
    n = len(Y)
    fit_result = estimator.fit(X, Y)
    if not fit_result['converged']:
        return np.nan, np.nan, np.nan, np.nan

    theta = fit_result['theta'][param_index]
    se = fit_result['se_model'][param_index]
    dof = fit_result['dof']

    if dof > 0:
        t_val = stats.t.ppf(1 - alpha / 2, dof)
    else:
        t_val = stats.norm.ppf(1 - alpha / 2)

    return theta, se, theta - t_val * se, theta + t_val * se


def robust_inference(estimator, X, Y, param_index=1, alpha=0.05):
    """Inference using robust/sandwich SE only (no adaptation)."""
    n = len(Y)
    fit_result = estimator.fit(X, Y)
    if not fit_result['converged']:
        return np.nan, np.nan, np.nan, np.nan

    theta = fit_result['theta'][param_index]
    se = fit_result['se_robust'][param_index]
    dof = fit_result['dof']

    if dof > 0:
        t_val = stats.t.ppf(1 - alpha / 2, dof)
    else:
        t_val = stats.norm.ppf(1 - alpha / 2)

    return theta, se, theta - t_val * se, theta + t_val * se


# =============================================================================
# VALIDATION: Compare generalized OLS against existing AMRI v2
# =============================================================================

def validate_against_existing():
    """Verify OLSEstimator + amri_v2_general reproduces existing results."""
    print("=" * 60)
    print("VALIDATION: OLSEstimator vs existing AMRI v2")
    print("=" * 60)

    rng = np.random.default_rng(42)
    n = 200
    B = 5000

    estimator = OLSEstimator()

    # Test under correct specification
    coverages_general = []
    coverages_manual = []

    for b in range(B):
        X = rng.standard_normal(n)
        eps = rng.standard_normal(n)
        Y = 1.0 + X + eps

        # Generalized AMRI v2
        result = amri_v2_general(estimator, X, Y, param_index=1)
        covers_gen = (result['ci_lo'] <= 1.0) and (result['ci_hi'] >= 1.0)
        coverages_general.append(covers_gen)

        # Manual AMRI v2 (reimplementation for validation)
        Xa = sm.add_constant(X)
        m_naive = sm.OLS(Y, Xa).fit()
        m_robust = sm.OLS(Y, Xa).fit(cov_type='HC3')
        se_n = m_naive.bse[1]
        se_r = m_robust.bse[1]
        ratio = se_r / max(se_n, 1e-10)
        log_r = abs(np.log(ratio))
        lo = 1.0 / np.sqrt(n)
        hi = 2.0 / np.sqrt(n)
        w = float(np.clip((log_r - lo) / (hi - lo), 0, 1))
        se_m = (1 - w) * se_n + w * se_r
        t_val = stats.t.ppf(0.975, n - 2)
        theta = m_naive.params[1]
        ci_lo_m = theta - t_val * se_m
        ci_hi_m = theta + t_val * se_m
        covers_man = (ci_lo_m <= 1.0) and (ci_hi_m >= 1.0)
        coverages_manual.append(covers_man)

    cov_gen = np.mean(coverages_general)
    cov_man = np.mean(coverages_manual)

    print(f"  Generalized AMRI v2 coverage: {cov_gen:.4f}")
    print(f"  Manual AMRI v2 coverage:      {cov_man:.4f}")
    print(f"  Difference:                   {abs(cov_gen - cov_man):.4f}")
    print(f"  Match: {'YES' if abs(cov_gen - cov_man) < 0.01 else 'NO'}")
    print()

    return cov_gen, cov_man


# =============================================================================
# QUICK DEMO: AMRI v2 across estimator types
# =============================================================================

def demo_all_estimators():
    """Demonstrate AMRI v2 works with all four estimator types."""
    print("=" * 60)
    print("DEMO: AMRI v2 across estimator types")
    print("=" * 60)

    rng = np.random.default_rng(2026)
    n = 500

    # 1. Simple OLS (correct spec)
    print("\n--- OLS (correct specification) ---")
    X = rng.standard_normal(n)
    Y = 1.0 + 0.5 * X + rng.standard_normal(n)
    est = OLSEstimator()
    res = amri_v2_general(est, X, Y, param_index=1)
    print(f"  theta={res['theta']:.4f}, se={res['se']:.4f}, "
          f"w={res['w']:.4f}, CI=[{res['ci_lo']:.4f}, {res['ci_hi']:.4f}]")
    print(f"  SE_model={res['se_model']:.4f}, SE_robust={res['se_robust']:.4f}, "
          f"ratio={res['ratio']:.4f}")

    # 2. Multiple OLS (p=3, heteroscedastic)
    print("\n--- Multiple OLS (p=3, heteroscedastic) ---")
    X_multi = rng.standard_normal((n, 3))
    sd = np.exp(0.5 * X_multi[:, 0])
    Y = 1.0 + 0.5 * X_multi[:, 0] + 0.3 * X_multi[:, 1] - 0.2 * X_multi[:, 2] \
        + rng.standard_normal(n) * sd
    est = MultipleOLSEstimator()
    res = amri_v2_general(est, X_multi, Y, param_index=1)
    print(f"  theta={res['theta']:.4f}, se={res['se']:.4f}, "
          f"w={res['w']:.4f}, CI=[{res['ci_lo']:.4f}, {res['ci_hi']:.4f}]")
    print(f"  SE_model={res['se_model']:.4f}, SE_robust={res['se_robust']:.4f}, "
          f"ratio={res['ratio']:.4f}")

    # 3. Logistic regression (correct spec)
    print("\n--- Logistic Regression (correct specification) ---")
    X_log = rng.standard_normal(n)
    logit_p = -0.5 + 1.0 * X_log
    prob = 1.0 / (1.0 + np.exp(-logit_p))
    Y_bin = rng.binomial(1, prob)
    est = LogisticEstimator()
    res = amri_v2_general(est, X_log, Y_bin, param_index=1)
    print(f"  theta={res['theta']:.4f}, se={res['se']:.4f}, "
          f"w={res['w']:.4f}, CI=[{res['ci_lo']:.4f}, {res['ci_hi']:.4f}]")
    print(f"  SE_model={res['se_model']:.4f}, SE_robust={res['se_robust']:.4f}, "
          f"ratio={res['ratio']:.4f}")

    # 4. Poisson regression (overdispersed)
    print("\n--- Poisson Regression (overdispersed) ---")
    X_pois = rng.standard_normal(n)
    log_mu = 0.5 + 0.3 * X_pois
    mu = np.exp(log_mu)
    # Overdispersed: use NegBin instead of Poisson
    Y_count = rng.negative_binomial(n=5, p=5.0 / (5.0 + mu))
    est = PoissonEstimator()
    res = amri_v2_general(est, X_pois, Y_count, param_index=1)
    print(f"  theta={res['theta']:.4f}, se={res['se']:.4f}, "
          f"w={res['w']:.4f}, CI=[{res['ci_lo']:.4f}, {res['ci_hi']:.4f}]")
    print(f"  SE_model={res['se_model']:.4f}, SE_robust={res['se_robust']:.4f}, "
          f"ratio={res['ratio']:.4f}")

    print()


if __name__ == '__main__':
    validate_against_existing()
    demo_all_estimators()
