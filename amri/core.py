"""
Core AMRI inference functions.

This module implements the Adaptive Misspecification-Robust Inference (AMRI)
algorithm in two versions:

- **v2** (recommended): Soft-threshold blending with tunable constants ``c1``
  and ``c2``.  The blending weight varies smoothly from 0 to 1 as the SE
  ratio departs from unity, yielding near-minimax optimal coverage.

- **v1**: Hard-switching variant that toggles between model-based and robust
  standard errors at a fixed threshold ``tau = 1 + 2 / sqrt(n)``.

The high-level entry point :func:`adaptive_ci` fits an OLS model internally
and returns an :class:`AMRIResult` with the blended confidence interval.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AMRIResult:
    """Container for AMRI inference results.

    Attributes
    ----------
    theta : float
        Point estimate (coefficient of interest).
    se : float
        AMRI-blended standard error used for the confidence interval.
    ci_low : float
        Lower bound of the ``(1 - alpha)`` confidence interval.
    ci_high : float
        Upper bound of the ``(1 - alpha)`` confidence interval.
    w : float
        Blending weight in ``[0, 1]``.  ``w = 0`` corresponds to the
        efficient (model-based) SE; ``w = 1`` to the robust (sandwich) SE.
    se_model : float
        Model-based (naive) standard error.
    se_robust : float
        Heteroskedasticity-robust (sandwich) standard error.
    ratio : float
        SE ratio ``R = se_robust / se_model``.
    method : str
        Version string, either ``'v1'`` or ``'v2'``.
    dof : Optional[int]
        Residual degrees of freedom (``n - p`` for OLS).  When ``None``,
        Gaussian critical values are used instead of the *t*-distribution.
    """

    theta: float
    se: float
    ci_low: float
    ci_high: float
    w: float
    se_model: float
    se_robust: float
    ratio: float
    method: str
    dof: Optional[int]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _critical_value(alpha: float, dof: Optional[int]) -> float:
    """Return the two-sided critical value at level ``alpha``.

    Uses the *t*-distribution when *dof* is provided, otherwise the
    standard normal.
    """
    if dof is not None and dof > 0:
        return float(stats.t.ppf(1.0 - alpha / 2.0, df=dof))
    return float(stats.norm.ppf(1.0 - alpha / 2.0))


# ---------------------------------------------------------------------------
# AMRI v2 — soft-threshold blending
# ---------------------------------------------------------------------------

def amri_v2(
    se_naive: float,
    se_robust: float,
    n: int,
    alpha: float = 0.05,
    theta: Optional[float] = None,
    dof: Optional[int] = None,
    c1: float = 1.0,
    c2: float = 2.0,
) -> AMRIResult:
    """AMRI v2: soft-threshold blending of model-based and robust SEs.

    Parameters
    ----------
    se_naive : float
        Model-based (naive) standard error.
    se_robust : float
        Heteroskedasticity-robust (sandwich) standard error.
    n : int
        Sample size, used to scale the blending thresholds.
    alpha : float, optional
        Significance level for the confidence interval (default 0.05).
    theta : float or None, optional
        Point estimate.  If ``None``, the CI bounds are returned as offsets
        from zero (+/- half-width).
    dof : int or None, optional
        Residual degrees of freedom.  When ``None`` the standard normal
        quantile is used.
    c1 : float, optional
        Lower blending constant (default 1.0).  Below ``c1 / sqrt(n)``
        the weight is zero (fully efficient).
    c2 : float, optional
        Upper blending constant (default 2.0).  Above ``c2 / sqrt(n)``
        the weight is one (fully robust).

    Returns
    -------
    AMRIResult
        Dataclass with the blended CI and diagnostics.

    Notes
    -----
    The blending rule is::

        R  = se_robust / se_naive
        s  = |log(R)|
        lo = c1 / sqrt(n)
        hi = c2 / sqrt(n)
        w  = clip((s - lo) / (hi - lo), 0, 1)
        SE = (1 - w) * se_naive + w * se_robust
    """
    if se_naive <= 0:
        raise ValueError("se_naive must be positive")
    if se_robust <= 0:
        raise ValueError("se_robust must be positive")
    if n < 2:
        raise ValueError("n must be at least 2")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if c1 < 0 or c2 <= c1:
        raise ValueError("Need 0 <= c1 < c2")

    R = se_robust / se_naive
    s = abs(math.log(R))

    sqrt_n = math.sqrt(n)
    lo = c1 / sqrt_n
    hi = c2 / sqrt_n

    # Blending weight
    if hi == lo:
        w = 0.0 if s <= lo else 1.0
    else:
        w = float(np.clip((s - lo) / (hi - lo), 0.0, 1.0))

    se_amri = (1.0 - w) * se_naive + w * se_robust
    cv = _critical_value(alpha, dof)

    theta_val = theta if theta is not None else 0.0
    ci_low = theta_val - cv * se_amri
    ci_high = theta_val + cv * se_amri

    return AMRIResult(
        theta=theta_val,
        se=se_amri,
        ci_low=ci_low,
        ci_high=ci_high,
        w=w,
        se_model=se_naive,
        se_robust=se_robust,
        ratio=R,
        method="v2",
        dof=dof,
    )


# ---------------------------------------------------------------------------
# AMRI v1 — hard switching
# ---------------------------------------------------------------------------

def amri_v1(
    se_naive: float,
    se_robust: float,
    n: int,
    alpha: float = 0.05,
    theta: Optional[float] = None,
    dof: Optional[int] = None,
) -> AMRIResult:
    """AMRI v1: hard-switching between model-based and robust SEs.

    Parameters
    ----------
    se_naive : float
        Model-based (naive) standard error.
    se_robust : float
        Heteroskedasticity-robust (sandwich) standard error.
    n : int
        Sample size.
    alpha : float, optional
        Significance level (default 0.05).
    theta : float or None, optional
        Point estimate.
    dof : int or None, optional
        Residual degrees of freedom.

    Returns
    -------
    AMRIResult

    Notes
    -----
    The switching rule is::

        R   = se_robust / se_naive
        tau = 1 + 2 / sqrt(n)
        if R > tau  or  R < 1/tau:
            SE = 1.05 * se_robust    # robust mode
        else:
            SE = se_naive            # efficient mode
    """
    if se_naive <= 0:
        raise ValueError("se_naive must be positive")
    if se_robust <= 0:
        raise ValueError("se_robust must be positive")
    if n < 2:
        raise ValueError("n must be at least 2")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    R = se_robust / se_naive
    tau = 1.0 + 2.0 / math.sqrt(n)

    if R > tau or R < 1.0 / tau:
        se_used = 1.05 * se_robust
        w = 1.0
    else:
        se_used = se_naive
        w = 0.0

    cv = _critical_value(alpha, dof)
    theta_val = theta if theta is not None else 0.0
    ci_low = theta_val - cv * se_used
    ci_high = theta_val + cv * se_used

    return AMRIResult(
        theta=theta_val,
        se=se_used,
        ci_low=ci_low,
        ci_high=ci_high,
        w=w,
        se_model=se_naive,
        se_robust=se_robust,
        ratio=R,
        method="v1",
        dof=dof,
    )


# ---------------------------------------------------------------------------
# High-level convenience function
# ---------------------------------------------------------------------------

def adaptive_ci(
    X: ArrayLike,
    y: ArrayLike,
    alpha: float = 0.05,
    c1: float = 1.0,
    c2: float = 2.0,
    param_index: int = 1,
    version: str = "v2",
) -> AMRIResult:
    """Compute an AMRI confidence interval for an OLS coefficient.

    This is the primary user-facing entry point.  It fits an OLS model to
    ``(X, y)`` with an automatically-added intercept, extracts model-based
    and HC3 robust standard errors for the coefficient at *param_index*,
    and returns the blended confidence interval.

    Parameters
    ----------
    X : array_like, shape ``(n,)`` or ``(n, p)``
        Predictor matrix **without** an intercept column.  A 1-D array is
        treated as a single predictor.
    y : array_like, shape ``(n,)``
        Response vector.
    alpha : float, optional
        Significance level (default 0.05 for a 95 % CI).
    c1 : float, optional
        Lower blending constant (v2 only; default 1.0).
    c2 : float, optional
        Upper blending constant (v2 only; default 2.0).
    param_index : int, optional
        Index of the coefficient of interest in the design matrix
        **including** the intercept column (default 1, i.e. the first
        predictor).
    version : ``'v1'`` or ``'v2'``, optional
        Which AMRI variant to use (default ``'v2'``).

    Returns
    -------
    AMRIResult
        Dataclass with the blended CI, blending weight, and diagnostics.

    Raises
    ------
    ValueError
        If *version* is not ``'v1'`` or ``'v2'``, or if dimensions are
        incompatible.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> X = rng.standard_normal(200)
    >>> y = 1.0 + 2.0 * X + rng.standard_normal(200)
    >>> result = adaptive_ci(X, y)
    >>> print(f"theta = {result.theta:.3f}, w = {result.w:.3f}")
    """
    import statsmodels.api as sm

    # --- Input validation & shaping -----------------------------------------
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError("X must be 1-D or 2-D")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X has {X.shape[0]} rows but y has {y.shape[0]} elements"
        )

    n, p = X.shape

    # Add intercept
    X_design = sm.add_constant(X, has_constant="skip")
    n_params = X_design.shape[1]

    if not 0 <= param_index < n_params:
        raise ValueError(
            f"param_index={param_index} out of range for {n_params} parameters"
        )

    # --- Fit OLS ------------------------------------------------------------
    model = sm.OLS(y, X_design).fit()

    theta_hat = float(model.params[param_index])
    se_naive = float(model.bse[param_index])

    # HC3 robust standard errors
    robust_model = model.get_robustcov_results(cov_type="HC3")
    se_hc3 = float(robust_model.bse[param_index])

    dof = int(model.df_resid)

    # --- Dispatch to the chosen version -------------------------------------
    version = version.lower().strip()
    if version == "v2":
        return amri_v2(
            se_naive=se_naive,
            se_robust=se_hc3,
            n=n,
            alpha=alpha,
            theta=theta_hat,
            dof=dof,
            c1=c1,
            c2=c2,
        )
    elif version == "v1":
        return amri_v1(
            se_naive=se_naive,
            se_robust=se_hc3,
            n=n,
            alpha=alpha,
            theta=theta_hat,
            dof=dof,
        )
    else:
        raise ValueError(f"version must be 'v1' or 'v2', got '{version}'")
