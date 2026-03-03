"""
Simulation engine for the dashboard.
Wraps existing src/competitor_comparison.py functions — no reimplementation.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import DGP functions and method runner from existing code
from src.competitor_comparison import (
    DGP_FUNCTIONS,
    run_method as _run_method,
    amri_v2_ci,
    amri_v1_ci,
    aks_adaptive_ci,
    compute_hc4_se,
    compute_hc5_se,
)

# All available methods
ALL_METHODS = [
    "Naive_OLS",
    "Sandwich_HC3",
    "Sandwich_HC4",
    "Sandwich_HC5",
    "Pairs_Bootstrap",
    "Wild_Bootstrap",
    "Bootstrap_t",
    "AKS_Adaptive",
    "AMRI_v1",
    "AMRI_v2",
]

# DGP metadata for the UI
DGP_INFO = {
    "Heteroscedastic": {
        "label": "Heteroscedastic Errors",
        "desc": "Var(\u03b5|X) = exp(\u03b4\u00b7X). Variance misspecification.",
        "formula": "Y = X + exp(\u03b4\u00b7X/2)\u00b7\u03b5",
    },
    "Heavy_Tails": {
        "label": "Heavy-Tailed Errors",
        "desc": "Errors follow t-distribution with df = 10 \u2212 8\u03b4.",
        "formula": "Y = X + \u03b5,  \u03b5 ~ t(df)",
    },
    "Nonlinear": {
        "label": "Nonlinearity",
        "desc": "Quadratic term omitted from linear fit.",
        "formula": "Y = X + \u03b4\u00b7X\u00b2 + \u03b5",
    },
    "Omitted_Variable": {
        "label": "Omitted Variable",
        "desc": "Correlated regressor omitted (\u03c1 = 0.5).",
        "formula": "Y = X\u2081 + \u03b4\u00b7X\u2082 + \u03b5  (X\u2082 omitted)",
    },
    "Clustering": {
        "label": "Ignored Clustering",
        "desc": "Cluster random effects ignored.",
        "formula": "Y = X + \u221a\u03b4\u00b7u_g + \u03b5",
    },
    "Contaminated": {
        "label": "Contaminated Normal",
        "desc": "Mixture of normal and heavy-tailed errors.",
        "formula": "\u03b5 ~ (1\u2212p)\u00b7N(0,1) + p\u00b7N(0,25)",
    },
    "Custom_DGP": {
        "label": "Custom DGP (specify distribution)",
        "desc": "Custom error distribution + misspecification.",
        "formula": "Y = \u03b2\u2080 + \u03b2\u2081\u00b7X + \u03b4\u00b7f(X) + \u03c3\u00b7\u03b5",
    },
}

# Error distribution options for Custom DGP
ERROR_DISTRIBUTIONS = {
    "normal": {"label": "Normal / Gaussian  N(0,1)",
               "desc": "Standard Gaussian errors"},
    "t_heavy": {"label": "Student-t (heavy tails)",
                "desc": "Heavy-tailed (specify df)"},
    "contaminated": {"label": "Contaminated Normal",
                     "desc": "Mix: (1\u2212p)\u00b7N(0,1) + p\u00b7N(0, \u03c3\u00b2)"},
    "skewed": {"label": "Skew-Normal",
               "desc": "Asymmetric errors (specify skew)"},
    "uniform": {"label": "Uniform",
                "desc": "Bounded errors ~ U(\u22121, 1)"},
    "lognormal": {"label": "Log-Normal (centered)",
                  "desc": "Right-skewed errors, mean-centered"},
    "exponential": {"label": "Exponential (centered)",
                    "desc": "Right-skewed, rate=1, mean-centered"},
    "gamma": {"label": "Gamma (centered)",
              "desc": "Flexible shape (specify shape k)"},
    "poisson": {"label": "Poisson (centered)",
                "desc": "Count-like errors (specify \u03bb)"},
    "chi2": {"label": "Chi-Squared (centered)",
             "desc": "Right-skewed (specify df)"},
    "beta_sym": {"label": "Beta (symmetric, centered)",
                 "desc": "Bounded on [0,1], specify \u03b1=\u03b2"},
    "laplace": {"label": "Laplace (double exponential)",
                "desc": "Heavier tails than Normal, peaked"},
    "cauchy": {"label": "Cauchy (extreme heavy tails)",
               "desc": "No mean/variance \u2014 extreme outliers"},
    "weibull": {"label": "Weibull (centered)",
                "desc": "Flexible shape (specify k)"},
    "pareto": {"label": "Pareto (centered)",
               "desc": "Power-law tail (specify \u03b1)"},
    "mixture_bimodal": {"label": "Bimodal Mixture",
                        "desc": "0.5\u00b7N(\u22122,1) + 0.5\u00b7N(2,1)"},
}

# Misspecification type options for Custom DGP
MISSPEC_TYPES = {
    "none": {"label": "None (correctly specified)",
             "desc": "No misspecification, delta ignored"},
    "heteroscedastic": {"label": "Heteroscedastic (exponential)",
                        "desc": "Var(eps|X) = exp(delta*X)"},
    "heteroscedastic_linear": {"label": "Heteroscedastic (linear)",
                               "desc": "Var(eps|X) = (1 + delta*|X|)^2"},
    "nonlinear": {"label": "Nonlinear (quadratic)",
                  "desc": "Omitted delta*X^2 term"},
    "cubic": {"label": "Nonlinear (cubic)",
              "desc": "Omitted delta*X^3 term"},
    "omitted": {"label": "Omitted Variable",
                "desc": "Correlated Z omitted (rho=0.5)"},
    "interaction": {"label": "Omitted Interaction",
                    "desc": "Omitted delta*X*Z interaction term"},
    "clustering": {"label": "Ignored Clustering",
                   "desc": "Cluster random effects (20 groups)"},
    "measurement_error": {"label": "Measurement Error in X",
                          "desc": "X observed with noise, X* = X + delta*nu"},
    "threshold": {"label": "Threshold / Regime Switch",
                  "desc": "Slope changes by delta when X > 0"},
    "autocorrelated": {"label": "Autocorrelated Errors",
                       "desc": "AR(1) errors with rho = delta"},
    "breakpoint": {"label": "Structural Break",
                   "desc": "Slope shifts by delta at midpoint of X"},
    "skewed_effect": {"label": "Skewed Partial Effect",
                      "desc": "True effect is beta1 + delta*X (varies with X)"},
    "loglinear": {"label": "Log-Linear (exp link)",
                  "desc": "True Y = exp(beta0 + beta1*X + eps), fitted linear"},
}


def custom_dgp(n, delta, rng, error_dist="normal", error_params=None,
               misspec_type="heteroscedastic", beta0=0.0, beta1=1.0,
               sigma=1.0):
    """
    Fully customizable DGP for the dashboard.

    Parameters
    ----------
    error_dist : str
        One of: "normal", "t_heavy", "contaminated", "skewed",
                "uniform", "lognormal"
    error_params : dict
        Distribution-specific params: {"df": 3}, {"p": 0.1, "scale": 5},
        {"skew": 5}
    misspec_type : str
        One of: "none", "heteroscedastic", "nonlinear", "omitted"
    """
    if error_params is None:
        error_params = {}

    X = rng.standard_normal(n)

    # Generate errors from chosen distribution
    if error_dist == "t_heavy":
        df = error_params.get("df", 3)
        df = max(2.1, df)
        eps = rng.standard_t(df, n) / np.sqrt(df / (df - 2))
    elif error_dist == "contaminated":
        p = error_params.get("p", 0.1)
        scale = error_params.get("scale", 5.0)
        contam = rng.random(n) < p
        eps = rng.standard_normal(n)
        eps[contam] *= scale
    elif error_dist == "skewed":
        skew_val = error_params.get("skew", 5.0)
        raw = stats.skewnorm.rvs(skew_val, size=n, random_state=rng)
        eps = (raw - np.mean(raw)) / np.std(raw)
    elif error_dist == "uniform":
        eps = rng.uniform(-np.sqrt(3), np.sqrt(3), n)  # unit variance
    elif error_dist == "lognormal":
        raw = rng.lognormal(0, 0.5, n)
        eps = (raw - np.mean(raw)) / np.std(raw)
    elif error_dist == "exponential":
        raw = rng.exponential(1.0, n)
        eps = (raw - np.mean(raw)) / np.std(raw)
    elif error_dist == "gamma":
        shape_k = error_params.get("shape", 2.0)
        raw = rng.gamma(max(0.5, shape_k), 1.0, n)
        eps = (raw - np.mean(raw)) / np.std(raw)
    elif error_dist == "poisson":
        lam = error_params.get("lam", 5.0)
        raw = rng.poisson(max(1, lam), n).astype(float)
        eps = (raw - np.mean(raw)) / max(np.std(raw), 1e-10)
    elif error_dist == "chi2":
        df = error_params.get("df", 3)
        raw = rng.chisquare(max(1, df), n)
        eps = (raw - np.mean(raw)) / np.std(raw)
    elif error_dist == "beta_sym":
        ab = error_params.get("alpha_beta", 2.0)
        raw = rng.beta(max(0.5, ab), max(0.5, ab), n)
        eps = (raw - np.mean(raw)) / np.std(raw)
    elif error_dist == "laplace":
        eps = rng.laplace(0, 1 / np.sqrt(2), n)  # unit variance
    elif error_dist == "cauchy":
        raw = rng.standard_cauchy(n)
        # Trim extreme outliers to keep numerics stable (winsorize)
        p1, p99 = np.percentile(raw, [1, 99])
        raw = np.clip(raw, p1, p99)
        eps = (raw - np.median(raw)) / max(
            np.std(raw), 1e-10)
    elif error_dist == "weibull":
        shape_k = error_params.get("shape", 1.5)
        raw = rng.weibull(max(0.5, shape_k), n)
        eps = (raw - np.mean(raw)) / np.std(raw)
    elif error_dist == "pareto":
        a = error_params.get("alpha", 3.0)
        a = max(2.1, a)  # need a > 2 for finite variance
        raw = (rng.pareto(a, n) + 1)
        eps = (raw - np.mean(raw)) / np.std(raw)
    elif error_dist == "mixture_bimodal":
        comp = rng.choice(2, n)
        eps = np.where(comp == 0,
                       rng.normal(-2, 1, n),
                       rng.normal(2, 1, n))
        eps = (eps - np.mean(eps)) / np.std(eps)
    else:
        eps = rng.standard_normal(n)

    eps *= sigma

    # Apply misspecification
    if misspec_type == "heteroscedastic":
        var_mult = np.exp(delta * X / 2)
        eps = eps * var_mult
    elif misspec_type == "heteroscedastic_linear":
        var_mult = 1 + delta * np.abs(X)
        eps = eps * var_mult
    elif misspec_type == "nonlinear":
        Y = beta0 + beta1 * X + delta * X**2 + eps
        return X, Y, beta1
    elif misspec_type == "cubic":
        Y = beta0 + beta1 * X + delta * X**3 + eps
        return X, Y, beta1
    elif misspec_type == "omitted":
        rho = 0.5
        Z = rho * X + np.sqrt(1 - rho**2) * rng.standard_normal(n)
        Y = beta0 + beta1 * X + delta * Z + eps
        return X, Y, beta1 + delta * rho
    elif misspec_type == "interaction":
        Z = rng.standard_normal(n)
        Y = beta0 + beta1 * X + delta * X * Z + eps
        return X, Y, beta1
    elif misspec_type == "clustering":
        n_groups = 20
        groups = np.repeat(np.arange(n_groups), int(np.ceil(n / n_groups)))[:n]
        group_effects = rng.standard_normal(n_groups) * np.sqrt(delta)
        eps = eps + group_effects[groups]
    elif misspec_type == "measurement_error":
        # True X is X, but we observe X* = X + delta*noise
        X_true = X.copy()
        X = X_true + delta * rng.standard_normal(n)
        Y = beta0 + beta1 * X_true + eps
        # theta_true is still beta1, but OLS on X* is biased
        return X, Y, beta1
    elif misspec_type == "threshold":
        # Slope is beta1 for X<=0, beta1+delta for X>0
        slope = np.where(X > 0, beta1 + delta, beta1)
        Y = beta0 + slope * X + eps
        return X, Y, beta1  # linear fit targets beta1
    elif misspec_type == "autocorrelated":
        # AR(1) errors with rho = delta (capped at 0.99)
        rho_ar = min(delta, 0.99)
        ar_eps = np.zeros(n)
        ar_eps[0] = eps[0]
        for i in range(1, n):
            ar_eps[i] = rho_ar * ar_eps[i - 1] + eps[i]
        eps = ar_eps
    elif misspec_type == "breakpoint":
        # Structural break: slope shifts by delta for upper half of X
        median_x = np.median(X)
        slope = np.where(X > median_x, beta1 + delta, beta1)
        Y = beta0 + slope * X + eps
        return X, Y, beta1
    elif misspec_type == "skewed_effect":
        # True marginal effect varies: beta1 + delta*X
        Y = beta0 + (beta1 + delta * X) * X + eps
        # = beta0 + beta1*X + delta*X^2, theta_true = beta1
        return X, Y, beta1
    elif misspec_type == "loglinear":
        # True DGP is log-linear but we fit linear
        Y_latent = beta0 + beta1 * X + eps * 0.3  # scale down eps
        Y = np.exp(Y_latent * delta + (1 - delta) * 0)
        # For delta=0 -> Y=1 (no misspec), delta=1 -> full exp link
        Y = (1 - delta) * (beta0 + beta1 * X + eps) + delta * Y
        return X, Y, beta1

    Y = beta0 + beta1 * X + eps
    return X, Y, beta1


# ============================================================================
# REAL DATASET CATALOG
# ============================================================================

REAL_DATASETS = {
    "California_MedInc": {
        "label": "California Housing — MedInc -> Price",
        "source": "sklearn",
        "desc": "Median income predicting house price (n=20,640). Strong heteroscedasticity.",
        "loader": "_load_california_medinc",
    },
    "California_Rooms": {
        "label": "California Housing — AveRooms -> Price",
        "source": "sklearn",
        "desc": "Average rooms predicting house price. Extreme SE ratio (4.7x).",
        "loader": "_load_california_rooms",
    },
    "Diabetes_BMI": {
        "label": "Diabetes — BMI -> Disease Progression",
        "source": "sklearn",
        "desc": "Body mass index predicting diabetes progression (n=442).",
        "loader": "_load_diabetes_bmi",
    },
    "Diabetes_BP": {
        "label": "Diabetes — Blood Pressure -> Disease",
        "source": "sklearn",
        "desc": "Blood pressure predicting disease progression (n=442).",
        "loader": "_load_diabetes_bp",
    },
    "Iris_Sepal": {
        "label": "Iris — Sepal Length -> Sepal Width",
        "source": "sklearn",
        "desc": "Classic Fisher Iris (n=150). Mild misspecification.",
        "loader": "_load_iris_sepal",
    },
    "Iris_Petal": {
        "label": "Iris — Petal Length -> Petal Width",
        "source": "sklearn",
        "desc": "Strong linear relationship (n=150).",
        "loader": "_load_iris_petal",
    },
}


def _load_california_medinc():
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing()
    return data.data[:, 0], data.target, "MedInc", "HousePrice"


def _load_california_rooms():
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing()
    return data.data[:, 3], data.target, "AveRooms", "HousePrice"


def _load_diabetes_bmi():
    from sklearn.datasets import load_diabetes
    data = load_diabetes()
    return data.data[:, 2], data.target, "BMI", "DiseaseProgress"


def _load_diabetes_bp():
    from sklearn.datasets import load_diabetes
    data = load_diabetes()
    return data.data[:, 3], data.target, "BloodPressure", "DiseaseProgress"


def _load_iris_sepal():
    from sklearn.datasets import load_iris
    data = load_iris()
    return data.data[:, 0], data.data[:, 1], "SepalLength", "SepalWidth"


def _load_iris_petal():
    from sklearn.datasets import load_iris
    data = load_iris()
    return data.data[:, 2], data.data[:, 3], "PetalLength", "PetalWidth"


_DATASET_LOADERS = {
    "_load_california_medinc": _load_california_medinc,
    "_load_california_rooms": _load_california_rooms,
    "_load_diabetes_bmi": _load_diabetes_bmi,
    "_load_diabetes_bp": _load_diabetes_bp,
    "_load_iris_sepal": _load_iris_sepal,
    "_load_iris_petal": _load_iris_petal,
}


def load_real_dataset(dataset_key):
    """Load a real dataset by key. Returns (X, Y, x_name, y_name)."""
    info = REAL_DATASETS[dataset_key]
    loader_func = _DATASET_LOADERS[info["loader"]]
    return loader_func()


# ============================================================================
# CORE: Run methods on ANY (X, Y) data
# ============================================================================

def run_on_data(X, Y, methods, alpha=0.05, c1=1.0, c2=2.0,
                theta_true=None, B_boot=99, seed=42):
    """
    Run ALL selected methods on a given (X, Y) dataset.
    Works for synthetic DGP data, real data, or user-uploaded data.

    Parameters
    ----------
    X, Y : array-like
        Predictor and response.
    methods : list[str]
        Method names to run.
    theta_true : float or None
        If known (synthetic DGP), used for coverage checking.
        If None (real data), estimated via bootstrap ground truth.

    Returns
    -------
    results : list[dict]
        Per-method: method, theta_hat, se, ci_low, ci_high, covers_true, width, w
    se_ratio : float
        HC3/Naive SE ratio (diagnostic).
    intercept : float
        OLS intercept.
    ols_slope : float
        OLS slope estimate.
    """
    X = np.asarray(X, dtype=float).ravel()
    Y = np.asarray(Y, dtype=float).ravel()
    n = len(X)
    rng = np.random.default_rng(seed)

    # Fit OLS for diagnostics
    Xa = sm.add_constant(X)
    se_naive_raw = se_hc3_raw = se_ratio = 1.0
    intercept = 0.0
    ols_slope = 0.0
    try:
        ols = sm.OLS(Y, Xa).fit()
        ols_r = sm.OLS(Y, Xa).fit(cov_type="HC3")
        intercept = float(ols.params[0])
        ols_slope = float(ols.params[1])
        se_naive_raw = float(ols.bse[1])
        se_hc3_raw = float(ols_r.bse[1])
        se_ratio = se_hc3_raw / max(se_naive_raw, 1e-10)
    except Exception:
        intercept = float(np.mean(Y) - np.mean(X))

    results = []
    for method in methods:
        try:
            boot_seed = rng.integers(0, 2**31)

            if method == "AMRI_v2":
                theta, se, ci_lo, ci_hi, w = amri_v2_ci(
                    X, Y, alpha=alpha, c1=c1, c2=c2
                )
            elif method == "AMRI_v1":
                out = amri_v1_ci(X, Y, alpha=alpha)
                theta, se, ci_lo, ci_hi = out
                se_n = sm.OLS(Y, Xa).fit().bse[1]
                se_r = sm.OLS(Y, Xa).fit(cov_type="HC3").bse[1]
                ratio = se_r / max(se_n, 1e-10)
                threshold = 1 + 2 / np.sqrt(n)
                w = 1.0 if (ratio > threshold or ratio < 1 / threshold) else 0.0
            else:
                out = _run_method(method, X, Y, alpha,
                                  boot_seed=boot_seed, B_boot=B_boot)
                theta, se, ci_lo, ci_hi = out[:4]
                w = None

            covers = None
            if theta_true is not None and not np.isnan(ci_lo):
                covers = bool(ci_lo <= theta_true <= ci_hi)
            width = ci_hi - ci_lo if not np.isnan(ci_lo) else np.nan

            results.append({
                "method": method,
                "theta_hat": float(theta) if not np.isnan(theta) else None,
                "se": float(se) if not np.isnan(se) else None,
                "ci_low": float(ci_lo) if not np.isnan(ci_lo) else None,
                "ci_high": float(ci_hi) if not np.isnan(ci_hi) else None,
                "covers_true": covers,
                "width": float(width) if not np.isnan(width) else None,
                "w": float(w) if w is not None and not np.isnan(w) else None,
            })
        except Exception:
            results.append({
                "method": method,
                "theta_hat": None, "se": None,
                "ci_low": None, "ci_high": None,
                "covers_true": None, "width": None, "w": None,
            })

    return results, float(se_ratio), float(intercept), float(ols_slope)


def run_single_dataset(dgp_name, delta, n, methods, alpha=0.05,
                       c1=1.0, c2=2.0, seed=42, B_boot=99):
    """
    Generate ONE synthetic dataset and run ALL selected methods on it.
    """
    rng = np.random.default_rng(seed)
    dgp_func = DGP_FUNCTIONS[dgp_name]
    X, Y, theta_true = dgp_func(n, delta, rng)
    X_flat = X.ravel() if X.ndim > 1 else X

    results, se_ratio, intercept, _ = run_on_data(
        X_flat, Y, methods, alpha, c1, c2, theta_true, B_boot, seed
    )

    return results, X_flat.tolist(), Y.tolist(), float(theta_true), float(intercept), float(se_ratio)


def run_monte_carlo(dgp_name, delta, n, methods, B, alpha=0.05,
                    c1=1.0, c2=2.0, seed=42, B_boot=99,
                    progress_callback=None):
    """
    Run B replications and compute summary statistics per method.
    """
    rng = np.random.default_rng(seed)

    method_data = {m: {"thetas": [], "covers": [], "widths": [], "ses": []}
                   for m in methods}
    all_weights = []
    theta_true = None

    for rep in range(B):
        rep_seed = rng.integers(0, 2**31)
        results, _, _, t_true, _, _ = run_single_dataset(
            dgp_name, delta, n, methods, alpha, c1, c2, rep_seed, B_boot
        )
        if theta_true is None:
            theta_true = t_true

        for r in results:
            m = r["method"]
            if r["theta_hat"] is not None:
                method_data[m]["thetas"].append(r["theta_hat"])
                method_data[m]["covers"].append(r["covers_true"])
                method_data[m]["widths"].append(r["width"])
                method_data[m]["ses"].append(r["se"])
            if m == "AMRI_v2" and r["w"] is not None:
                all_weights.append(r["w"])

        if progress_callback:
            progress_callback(rep + 1, B)

    summary = []
    for m in methods:
        d = method_data[m]
        valid = len(d["thetas"])
        if valid == 0:
            summary.append({
                "method": m, "coverage": None, "avg_width": None,
                "bias": None, "rmse": None, "coverage_accuracy": None,
                "valid_reps": 0,
            })
            continue

        thetas = np.array(d["thetas"])
        covers = np.array(d["covers"])
        widths = np.array(d["widths"])

        cov = float(np.mean(covers))
        summary.append({
            "method": m,
            "coverage": cov,
            "coverage_mc_se": float(np.sqrt(cov * (1 - cov) / valid)),
            "avg_width": float(np.mean(widths)),
            "median_width": float(np.median(widths)),
            "bias": float(np.mean(thetas) - theta_true),
            "rmse": float(np.sqrt(np.mean((thetas - theta_true) ** 2))),
            "coverage_accuracy": float(abs(cov - (1 - alpha))),
            "valid_reps": valid,
        })

    return summary, all_weights
