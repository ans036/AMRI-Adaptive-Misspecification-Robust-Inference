"""
Diagnostic utilities for AMRI.

This module provides helper functions for inspecting the model-based vs.
robust SE ratio, computing the blending weight, and generating a
human-readable diagnostic report.

Functions
---------
- :func:`se_ratio` -- compute ``SE_HC3 / SE_naive`` for an OLS fit.
- :func:`blending_weight` -- compute the AMRI v2 blending weight ``w``
  from a pre-computed ratio and sample size.
- :func:`amri_diagnose` -- comprehensive diagnostic returning a dictionary
  with the ratio, weight, mode label, and a textual recommendation.
"""

from __future__ import annotations

import math
from typing import Dict, Union

import numpy as np
from numpy.typing import ArrayLike


# ---------------------------------------------------------------------------
# SE ratio
# ---------------------------------------------------------------------------

def se_ratio(
    X: ArrayLike,
    y: ArrayLike,
    param_index: int = 1,
) -> float:
    """Compute the HC3-to-OLS standard-error ratio for a single coefficient.

    Parameters
    ----------
    X : array_like, shape ``(n,)`` or ``(n, p)``
        Predictor matrix **without** intercept.
    y : array_like, shape ``(n,)``
        Response vector.
    param_index : int, optional
        Index of the coefficient of interest (after the intercept has been
        prepended).  Default is ``1``.

    Returns
    -------
    float
        ``R = SE_HC3 / SE_naive`` (always positive).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal(200)
    >>> y = 1.0 + 2.0 * X + rng.standard_normal(200)
    >>> R = se_ratio(X, y)
    >>> 0 < R
    True
    """
    import statsmodels.api as sm

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    X_design = sm.add_constant(X, has_constant="skip")

    if not 0 <= param_index < X_design.shape[1]:
        raise ValueError(
            f"param_index={param_index} out of range for "
            f"{X_design.shape[1]} parameters"
        )

    model = sm.OLS(y, X_design).fit()
    se_naive = float(model.bse[param_index])

    robust_model = model.get_robustcov_results(cov_type="HC3")
    se_hc3 = float(robust_model.bse[param_index])

    return se_hc3 / se_naive


# ---------------------------------------------------------------------------
# Blending weight
# ---------------------------------------------------------------------------

def blending_weight(
    R: float,
    n: int,
    c1: float = 1.0,
    c2: float = 2.0,
) -> float:
    """Compute the AMRI v2 blending weight from the SE ratio.

    Parameters
    ----------
    R : float
        Standard-error ratio ``SE_robust / SE_naive``.
    n : int
        Sample size.
    c1 : float, optional
        Lower threshold constant (default 1.0).
    c2 : float, optional
        Upper threshold constant (default 2.0).

    Returns
    -------
    float
        Blending weight ``w`` in ``[0, 1]``.

    Notes
    -----
    The formula is::

        s  = |log(R)|
        lo = c1 / sqrt(n)
        hi = c2 / sqrt(n)
        w  = clip((s - lo) / (hi - lo), 0, 1)
    """
    if R <= 0:
        raise ValueError("R must be positive")
    if n < 1:
        raise ValueError("n must be at least 1")
    if c2 <= c1:
        raise ValueError("Need c1 < c2")

    s = abs(math.log(R))
    sqrt_n = math.sqrt(n)
    lo = c1 / sqrt_n
    hi = c2 / sqrt_n

    if hi == lo:
        return 0.0 if s <= lo else 1.0

    return float(np.clip((s - lo) / (hi - lo), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Comprehensive diagnostic
# ---------------------------------------------------------------------------

def amri_diagnose(
    X: ArrayLike,
    y: ArrayLike,
    alpha: float = 0.05,
    c1: float = 1.0,
    c2: float = 2.0,
    param_index: int = 1,
) -> Dict[str, Union[float, int, str]]:
    """Run a comprehensive AMRI diagnostic on an OLS regression.

    Parameters
    ----------
    X : array_like
        Predictor matrix (without intercept).
    y : array_like
        Response vector.
    alpha : float, optional
        Significance level (default 0.05).
    c1 : float, optional
        Lower blending constant (default 1.0).
    c2 : float, optional
        Upper blending constant (default 2.0).
    param_index : int, optional
        Coefficient index of interest (default 1).

    Returns
    -------
    dict
        A dictionary with the following keys:

        - **ratio** (*float*): ``SE_HC3 / SE_naive``.
        - **w** (*float*): AMRI v2 blending weight.
        - **se_naive** (*float*): Model-based SE.
        - **se_hc3** (*float*): HC3 robust SE.
        - **mode** (*str*): One of ``'efficient'``, ``'blending'``, or
          ``'robust'``.
        - **n** (*int*): Sample size.
        - **recommendation** (*str*): Human-readable guidance.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal(200)
    >>> y = 1.0 + 2.0 * X + rng.standard_normal(200)
    >>> diag = amri_diagnose(X, y)
    >>> diag["mode"] in ("efficient", "blending", "robust")
    True
    """
    import statsmodels.api as sm

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n = X.shape[0]
    X_design = sm.add_constant(X, has_constant="skip")

    if not 0 <= param_index < X_design.shape[1]:
        raise ValueError(
            f"param_index={param_index} out of range for "
            f"{X_design.shape[1]} parameters"
        )

    model = sm.OLS(y, X_design).fit()
    se_naive = float(model.bse[param_index])

    robust_model = model.get_robustcov_results(cov_type="HC3")
    se_hc3 = float(robust_model.bse[param_index])

    R = se_hc3 / se_naive
    w = blending_weight(R, n, c1=c1, c2=c2)

    # Classify mode
    if w == 0.0:
        mode = "efficient"
    elif w == 1.0:
        mode = "robust"
    else:
        mode = "blending"

    # Build recommendation text
    if mode == "efficient":
        recommendation = (
            f"The SE ratio R = {R:.4f} is close to 1.  The model appears "
            f"well-specified.  AMRI uses the efficient (model-based) SE "
            f"(blending weight w = {w:.4f})."
        )
    elif mode == "robust":
        recommendation = (
            f"The SE ratio R = {R:.4f} deviates substantially from 1.  "
            f"There is evidence of misspecification or heteroskedasticity.  "
            f"AMRI uses the full robust (sandwich) SE "
            f"(blending weight w = {w:.4f})."
        )
    else:
        recommendation = (
            f"The SE ratio R = {R:.4f} shows moderate deviation from 1.  "
            f"AMRI blends model-based and robust SEs with weight "
            f"w = {w:.4f}.  Consider inspecting residual plots for "
            f"heteroskedasticity patterns."
        )

    return {
        "ratio": R,
        "w": w,
        "se_naive": se_naive,
        "se_hc3": se_hc3,
        "mode": mode,
        "n": n,
        "recommendation": recommendation,
    }
