"""
Estimator classes for AMRI.

Each estimator wraps a regression model and exposes a uniform :meth:`fit`
interface that returns a dictionary of the form::

    {
        "theta":     float,   # coefficient estimate
        "se_model":  float,   # model-based standard error
        "se_robust": float,   # sandwich (robust) standard error
        "dof":       int,     # residual degrees of freedom
        "converged": bool,    # whether the model converged
    }

This dictionary can be passed directly to :func:`amri.core.amri_v2` or
:func:`amri.core.amri_v1`.

Available estimators
--------------------
- :class:`OLSEstimator` -- ordinary least squares (model SE from OLS,
  robust SE from HC3).
- :class:`MultipleOLSEstimator` -- alias for :class:`OLSEstimator` (works
  identically for any number of predictors).
- :class:`LogisticEstimator` -- logistic regression via GLM (Binomial /
  Logit); robust SE from HC1.
- :class:`PoissonEstimator` -- Poisson regression via GLM (Poisson / Log);
  robust SE from HC1.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import ArrayLike


class AMRIEstimator(ABC):
    """Abstract base class for AMRI-compatible estimators.

    Subclasses must implement :meth:`fit`.

    Parameters
    ----------
    param_index : int, optional
        Index of the parameter of interest within the fitted coefficient
        vector (after the intercept has been prepended).  Default is ``1``
        (the first predictor).
    add_intercept : bool, optional
        Whether to automatically prepend an intercept column
        (default ``True``).
    """

    def __init__(
        self,
        param_index: int = 1,
        add_intercept: bool = True,
    ) -> None:
        self.param_index = param_index
        self.add_intercept = add_intercept

    @abstractmethod
    def fit(self, X: ArrayLike, y: ArrayLike) -> Dict[str, Any]:
        """Fit the model and return AMRI-relevant quantities.

        Parameters
        ----------
        X : array_like, shape ``(n,)`` or ``(n, p)``
            Predictor matrix **without** an intercept column.
        y : array_like, shape ``(n,)``
            Response vector.

        Returns
        -------
        dict
            Keys: ``theta``, ``se_model``, ``se_robust``, ``dof``,
            ``converged``.
        """

    # -- helpers --

    def _prepare_X(self, X: ArrayLike) -> np.ndarray:
        """Validate, reshape, and optionally add an intercept."""
        import statsmodels.api as sm

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError("X must be 1-D or 2-D")
        if self.add_intercept:
            X = sm.add_constant(X, has_constant="skip")
        return X

    def _validate_index(self, n_params: int) -> None:
        if not 0 <= self.param_index < n_params:
            raise ValueError(
                f"param_index={self.param_index} is out of range for "
                f"{n_params} parameters"
            )


# ---------------------------------------------------------------------------
# OLS
# ---------------------------------------------------------------------------

class OLSEstimator(AMRIEstimator):
    """Ordinary Least Squares estimator.

    Model-based SEs are taken from the standard OLS covariance matrix.
    Robust SEs use the HC3 heteroskedasticity-consistent estimator.

    Parameters
    ----------
    param_index : int, optional
        Coefficient index of interest (default 1).
    add_intercept : bool, optional
        Prepend an intercept column (default ``True``).
    robust_cov : str, optional
        Robust covariance type passed to statsmodels (default ``'HC3'``).

    Examples
    --------
    >>> import numpy as np
    >>> est = OLSEstimator()
    >>> X = np.arange(100, dtype=float)
    >>> y = 1.0 + 2.0 * X + np.random.default_rng(0).standard_normal(100)
    >>> info = est.fit(X, y)
    >>> info["converged"]
    True
    """

    def __init__(
        self,
        param_index: int = 1,
        add_intercept: bool = True,
        robust_cov: str = "HC3",
    ) -> None:
        super().__init__(param_index=param_index, add_intercept=add_intercept)
        self.robust_cov = robust_cov

    def fit(self, X: ArrayLike, y: ArrayLike) -> Dict[str, Any]:
        import statsmodels.api as sm

        X_design = self._prepare_X(X)
        y = np.asarray(y, dtype=float).ravel()
        self._validate_index(X_design.shape[1])

        model = sm.OLS(y, X_design).fit()

        idx = self.param_index
        theta = float(model.params[idx])
        se_model = float(model.bse[idx])

        robust_results = model.get_robustcov_results(cov_type=self.robust_cov)
        se_robust = float(robust_results.bse[idx])

        return {
            "theta": theta,
            "se_model": se_model,
            "se_robust": se_robust,
            "dof": int(model.df_resid),
            "converged": True,  # OLS always converges (closed-form)
        }


class MultipleOLSEstimator(OLSEstimator):
    """OLS estimator for multiple predictors.

    This is an alias for :class:`OLSEstimator` -- it works identically
    regardless of the number of predictors.
    """

    pass


# ---------------------------------------------------------------------------
# Logistic regression (GLM Binomial / Logit)
# ---------------------------------------------------------------------------

class LogisticEstimator(AMRIEstimator):
    """Logistic regression estimator (Binomial GLM with logit link).

    Model-based SEs come from the Fisher information matrix.  Robust SEs
    use the HC1 sandwich estimator.

    Parameters
    ----------
    param_index : int, optional
        Coefficient index (default 1).
    add_intercept : bool, optional
        Prepend an intercept (default ``True``).
    robust_cov : str, optional
        Robust covariance type (default ``'HC1'``).
    max_iter : int, optional
        Maximum IRLS iterations (default 100).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> X = rng.standard_normal(200)
    >>> y = (1.0 + 0.5 * X + rng.logistic(size=200) > 0).astype(float)
    >>> est = LogisticEstimator()
    >>> info = est.fit(X, y)
    """

    def __init__(
        self,
        param_index: int = 1,
        add_intercept: bool = True,
        robust_cov: str = "HC1",
        max_iter: int = 100,
    ) -> None:
        super().__init__(param_index=param_index, add_intercept=add_intercept)
        self.robust_cov = robust_cov
        self.max_iter = max_iter

    def fit(self, X: ArrayLike, y: ArrayLike) -> Dict[str, Any]:
        import statsmodels.api as sm

        X_design = self._prepare_X(X)
        y = np.asarray(y, dtype=float).ravel()
        self._validate_index(X_design.shape[1])

        family = sm.families.Binomial(link=sm.families.links.Logit())

        try:
            model = sm.GLM(y, X_design, family=family).fit(
                maxiter=self.max_iter, disp=False
            )
            converged = model.converged
        except Exception:
            # Return NaN-filled result on convergence failure
            return {
                "theta": float("nan"),
                "se_model": float("nan"),
                "se_robust": float("nan"),
                "dof": int(X_design.shape[0] - X_design.shape[1]),
                "converged": False,
            }

        idx = self.param_index
        theta = float(model.params[idx])
        se_model = float(model.bse[idx])

        # Robust SEs
        try:
            robust_results = model.get_robustcov_results(
                cov_type=self.robust_cov
            )
            se_robust = float(robust_results.bse[idx])
        except Exception:
            se_robust = se_model  # graceful fallback

        dof = int(model.df_resid)

        return {
            "theta": theta,
            "se_model": se_model,
            "se_robust": se_robust,
            "dof": dof,
            "converged": bool(converged),
        }


# ---------------------------------------------------------------------------
# Poisson regression (GLM Poisson / Log)
# ---------------------------------------------------------------------------

class PoissonEstimator(AMRIEstimator):
    """Poisson regression estimator (GLM with log link).

    Model-based SEs come from the Fisher information matrix.  Robust SEs
    use the HC1 sandwich estimator.

    Parameters
    ----------
    param_index : int, optional
        Coefficient index (default 1).
    add_intercept : bool, optional
        Prepend an intercept (default ``True``).
    robust_cov : str, optional
        Robust covariance type (default ``'HC1'``).
    max_iter : int, optional
        Maximum IRLS iterations (default 100).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> X = rng.standard_normal(200)
    >>> y = rng.poisson(lam=np.exp(0.5 + 0.3 * X))
    >>> est = PoissonEstimator()
    >>> info = est.fit(X, y)
    """

    def __init__(
        self,
        param_index: int = 1,
        add_intercept: bool = True,
        robust_cov: str = "HC1",
        max_iter: int = 100,
    ) -> None:
        super().__init__(param_index=param_index, add_intercept=add_intercept)
        self.robust_cov = robust_cov
        self.max_iter = max_iter

    def fit(self, X: ArrayLike, y: ArrayLike) -> Dict[str, Any]:
        import statsmodels.api as sm

        X_design = self._prepare_X(X)
        y = np.asarray(y, dtype=float).ravel()
        self._validate_index(X_design.shape[1])

        family = sm.families.Poisson(link=sm.families.links.Log())

        try:
            model = sm.GLM(y, X_design, family=family).fit(
                maxiter=self.max_iter, disp=False
            )
            converged = model.converged
        except Exception:
            return {
                "theta": float("nan"),
                "se_model": float("nan"),
                "se_robust": float("nan"),
                "dof": int(X_design.shape[0] - X_design.shape[1]),
                "converged": False,
            }

        idx = self.param_index
        theta = float(model.params[idx])
        se_model = float(model.bse[idx])

        try:
            robust_results = model.get_robustcov_results(
                cov_type=self.robust_cov
            )
            se_robust = float(robust_results.bse[idx])
        except Exception:
            se_robust = se_model

        dof = int(model.df_resid)

        return {
            "theta": theta,
            "se_model": se_model,
            "se_robust": se_robust,
            "dof": dof,
            "converged": bool(converged),
        }
