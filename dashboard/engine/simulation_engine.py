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
        "desc": "Var(e|X) = exp(delta*X). Variance misspecification.",
        "formula": "Y = X + exp(delta*X/2) * e",
    },
    "Heavy_Tails": {
        "label": "Heavy-Tailed Errors",
        "desc": "Errors follow t-distribution with df = 10-8*delta.",
        "formula": "Y = X + e,  e ~ t(df)",
    },
    "Nonlinear": {
        "label": "Nonlinearity",
        "desc": "Quadratic term omitted from linear fit.",
        "formula": "Y = X + delta*X^2 + e",
    },
    "Omitted_Variable": {
        "label": "Omitted Variable",
        "desc": "Correlated regressor omitted (rho=0.5).",
        "formula": "Y = X1 + delta*X2 + e  (X2 omitted)",
    },
    "Clustering": {
        "label": "Ignored Clustering",
        "desc": "Cluster random effects ignored.",
        "formula": "Y = X + sqrt(delta)*u_g + e",
    },
    "Contaminated": {
        "label": "Contaminated Normal",
        "desc": "Mixture of normal and heavy-tailed errors.",
        "formula": "e ~ (1-p)*N(0,1) + p*N(0,25)",
    },
}


def run_single_dataset(dgp_name, delta, n, methods, alpha=0.05,
                       c1=1.0, c2=2.0, seed=42, B_boot=199):
    """
    Generate ONE dataset and run ALL selected methods on it.

    Returns
    -------
    results : list[dict]
        One dict per method with keys:
        method, theta_hat, se, ci_low, ci_high, covers_true, width, w
    X : ndarray
        Predictor values (for scatter plot)
    Y : ndarray
        Response values (for scatter plot)
    theta_true : float
        True parameter value
    intercept : float
        Fitted OLS intercept
    """
    rng = np.random.default_rng(seed)
    dgp_func = DGP_FUNCTIONS[dgp_name]
    X, Y, theta_true = dgp_func(n, delta, rng)

    # Flatten X if 2D
    X_flat = X.ravel() if X.ndim > 1 else X

    # Get the OLS intercept and SE ratio for diagnostics
    Xa = sm.add_constant(X_flat)
    se_naive_raw = se_hc3_raw = se_ratio = 1.0
    try:
        ols = sm.OLS(Y, Xa).fit()
        ols_r = sm.OLS(Y, Xa).fit(cov_type="HC3")
        intercept = ols.params[0]
        se_naive_raw = float(ols.bse[1])
        se_hc3_raw = float(ols_r.bse[1])
        se_ratio = se_hc3_raw / max(se_naive_raw, 1e-10)
    except Exception:
        intercept = np.mean(Y) - np.mean(X_flat)

    results = []
    for method in methods:
        try:
            boot_seed = rng.integers(0, 2**31)

            if method == "AMRI_v2":
                theta, se, ci_lo, ci_hi, w = amri_v2_ci(
                    X_flat, Y, alpha=alpha, c1=c1, c2=c2
                )
            elif method == "AMRI_v1":
                out = amri_v1_ci(X_flat, Y, alpha=alpha)
                theta, se, ci_lo, ci_hi = out
                # Compute w for display
                se_n = sm.OLS(Y, Xa).fit().bse[1]
                se_r = sm.OLS(Y, Xa).fit(cov_type="HC3").bse[1]
                ratio = se_r / max(se_n, 1e-10)
                threshold = 1 + 2 / np.sqrt(n)
                w = 1.0 if (ratio > threshold or ratio < 1 / threshold) else 0.0
            else:
                out = _run_method(method, X_flat, Y, alpha,
                                  boot_seed=boot_seed, B_boot=B_boot)
                theta, se, ci_lo, ci_hi = out[:4]
                w = None

            covers = bool(ci_lo <= theta_true <= ci_hi) if not np.isnan(ci_lo) else None
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
        except Exception as e:
            results.append({
                "method": method,
                "theta_hat": None, "se": None,
                "ci_low": None, "ci_high": None,
                "covers_true": None, "width": None, "w": None,
            })

    return results, X_flat.tolist(), Y.tolist(), float(theta_true), float(intercept), float(se_ratio)


def run_monte_carlo(dgp_name, delta, n, methods, B, alpha=0.05,
                    c1=1.0, c2=2.0, seed=42, B_boot=199,
                    progress_callback=None):
    """
    Run B replications and compute summary statistics per method.

    Returns
    -------
    summary : list[dict]
        Per-method summary: coverage, avg_width, bias, rmse, etc.
    all_weights : list[float]
        AMRI v2 blending weights across all reps (for histogram).
    """
    rng = np.random.default_rng(seed)

    # Storage: method -> lists of values
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

    # Compute summaries
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
