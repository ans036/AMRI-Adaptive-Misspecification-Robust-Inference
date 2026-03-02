"""
Tests for amri.estimators — OLS, Logistic, and Poisson estimators.
"""

import math

import numpy as np
import pytest

from amri.estimators import (
    LogisticEstimator,
    MultipleOLSEstimator,
    OLSEstimator,
    PoissonEstimator,
)


# ---------------------------------------------------------------------------
# OLSEstimator
# ---------------------------------------------------------------------------

class TestOLSEstimator:
    """Unit tests for OLSEstimator."""

    @pytest.fixture()
    def ols_data(self):
        rng = np.random.default_rng(42)
        n = 200
        X = rng.standard_normal(n)
        y = 1.0 + 2.0 * X + rng.standard_normal(n)
        return X, y

    def test_returns_required_keys(self, ols_data):
        X, y = ols_data
        est = OLSEstimator()
        info = est.fit(X, y)
        for key in ("theta", "se_model", "se_robust", "dof", "converged"):
            assert key in info, f"Missing key: {key}"

    def test_converged_always_true(self, ols_data):
        X, y = ols_data
        info = OLSEstimator().fit(X, y)
        assert info["converged"] is True

    def test_theta_near_true_value(self, ols_data):
        X, y = ols_data
        info = OLSEstimator().fit(X, y)
        assert abs(info["theta"] - 2.0) < 0.5, "Slope estimate too far from 2"

    def test_se_positive(self, ols_data):
        X, y = ols_data
        info = OLSEstimator().fit(X, y)
        assert info["se_model"] > 0
        assert info["se_robust"] > 0

    def test_dof_correct(self, ols_data):
        X, y = ols_data
        info = OLSEstimator().fit(X, y)
        # n=200, p=2 (intercept + X) => dof = 198
        assert info["dof"] == 198

    def test_multiple_predictors(self):
        rng = np.random.default_rng(7)
        n = 300
        X = rng.standard_normal((n, 3))
        y = 1.0 + X @ np.array([2.0, -1.0, 0.5]) + rng.standard_normal(n)
        est = OLSEstimator(param_index=2)
        info = est.fit(X, y)
        assert abs(info["theta"] - (-1.0)) < 0.5

    def test_param_index_zero_is_intercept(self, ols_data):
        X, y = ols_data
        est = OLSEstimator(param_index=0)
        info = est.fit(X, y)
        # Intercept should be near 1.0
        assert abs(info["theta"] - 1.0) < 0.5

    def test_invalid_param_index_raises(self, ols_data):
        X, y = ols_data
        est = OLSEstimator(param_index=10)
        with pytest.raises(ValueError, match="param_index"):
            est.fit(X, y)


# ---------------------------------------------------------------------------
# MultipleOLSEstimator (alias)
# ---------------------------------------------------------------------------

class TestMultipleOLSEstimator:
    def test_is_subclass_of_ols(self):
        assert issubclass(MultipleOLSEstimator, OLSEstimator)

    def test_fit_works(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 2))
        y = 1.0 + X @ np.array([1.0, -0.5]) + rng.standard_normal(100)
        info = MultipleOLSEstimator().fit(X, y)
        assert info["converged"] is True
        assert "theta" in info


# ---------------------------------------------------------------------------
# LogisticEstimator
# ---------------------------------------------------------------------------

class TestLogisticEstimator:
    """Unit tests for LogisticEstimator."""

    @pytest.fixture()
    def binary_data(self):
        rng = np.random.default_rng(42)
        n = 300
        X = rng.standard_normal(n)
        logit_p = 0.5 + 1.0 * X
        p = 1.0 / (1.0 + np.exp(-logit_p))
        y = rng.binomial(1, p)
        return X, y

    def test_returns_required_keys(self, binary_data):
        X, y = binary_data
        est = LogisticEstimator()
        info = est.fit(X, y)
        for key in ("theta", "se_model", "se_robust", "dof", "converged"):
            assert key in info

    def test_converged(self, binary_data):
        X, y = binary_data
        info = LogisticEstimator().fit(X, y)
        assert info["converged"] is True

    def test_se_positive(self, binary_data):
        X, y = binary_data
        info = LogisticEstimator().fit(X, y)
        assert info["se_model"] > 0
        assert info["se_robust"] > 0

    def test_theta_direction(self, binary_data):
        """The logistic coefficient should be positive (true beta=1.0)."""
        X, y = binary_data
        info = LogisticEstimator().fit(X, y)
        assert info["theta"] > 0

    def test_dof_positive(self, binary_data):
        X, y = binary_data
        info = LogisticEstimator().fit(X, y)
        assert info["dof"] > 0

    def test_perfect_separation_handling(self):
        """Perfect separation may cause convergence issues; should not crash."""
        X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=float)
        est = LogisticEstimator()
        info = est.fit(X, y)
        # Should return something (possibly with converged=False)
        assert "theta" in info


# ---------------------------------------------------------------------------
# PoissonEstimator
# ---------------------------------------------------------------------------

class TestPoissonEstimator:
    """Unit tests for PoissonEstimator."""

    @pytest.fixture()
    def count_data(self):
        rng = np.random.default_rng(42)
        n = 300
        X = rng.standard_normal(n)
        lam = np.exp(0.5 + 0.3 * X)
        y = rng.poisson(lam)
        return X, y

    def test_returns_required_keys(self, count_data):
        X, y = count_data
        est = PoissonEstimator()
        info = est.fit(X, y)
        for key in ("theta", "se_model", "se_robust", "dof", "converged"):
            assert key in info

    def test_converged(self, count_data):
        X, y = count_data
        info = PoissonEstimator().fit(X, y)
        assert info["converged"] is True

    def test_se_positive(self, count_data):
        X, y = count_data
        info = PoissonEstimator().fit(X, y)
        assert info["se_model"] > 0
        assert info["se_robust"] > 0

    def test_theta_direction(self, count_data):
        """The Poisson slope should be positive (true beta=0.3)."""
        X, y = count_data
        info = PoissonEstimator().fit(X, y)
        assert info["theta"] > 0

    def test_dof_positive(self, count_data):
        X, y = count_data
        info = PoissonEstimator().fit(X, y)
        assert info["dof"] > 0

    def test_zero_counts_ok(self):
        """Should handle data with many zeros."""
        rng = np.random.default_rng(99)
        n = 200
        X = rng.standard_normal(n)
        lam = np.exp(-2.0 + 0.1 * X)  # low rate => many zeros
        y = rng.poisson(lam)
        info = PoissonEstimator().fit(X, y)
        assert "theta" in info
        assert info["converged"] is True
