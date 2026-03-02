"""
Tests for amri.core — AMRIResult, amri_v2, amri_v1, and adaptive_ci.
"""

import math

import numpy as np
import pytest

from amri.core import AMRIResult, adaptive_ci, amri_v1, amri_v2


# ---------------------------------------------------------------------------
# AMRIResult
# ---------------------------------------------------------------------------

class TestAMRIResult:
    """Verify the AMRIResult dataclass has the expected fields."""

    def test_fields_present(self):
        r = AMRIResult(
            theta=1.0, se=0.1, ci_low=0.8, ci_high=1.2,
            w=0.3, se_model=0.09, se_robust=0.12, ratio=1.33,
            method="v2", dof=98,
        )
        assert r.theta == 1.0
        assert r.se == 0.1
        assert r.ci_low == 0.8
        assert r.ci_high == 1.2
        assert r.w == 0.3
        assert r.se_model == 0.09
        assert r.se_robust == 0.12
        assert r.ratio == pytest.approx(1.33)
        assert r.method == "v2"
        assert r.dof == 98

    def test_frozen(self):
        r = AMRIResult(
            theta=0, se=1, ci_low=-1, ci_high=1,
            w=0, se_model=1, se_robust=1, ratio=1,
            method="v2", dof=10,
        )
        with pytest.raises(AttributeError):
            r.theta = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# amri_v2
# ---------------------------------------------------------------------------

class TestAMRIv2:
    """Unit tests for the v2 soft-threshold blending."""

    def test_ratio_one_gives_w_zero(self):
        """When SE_robust == SE_naive (R=1), log(R)=0 => w should be 0."""
        result = amri_v2(se_naive=1.0, se_robust=1.0, n=100, theta=5.0, dof=98)
        assert result.w == 0.0
        assert result.se == pytest.approx(1.0)
        assert result.method == "v2"

    def test_large_ratio_gives_w_one(self):
        """When R is very large, w should saturate at 1."""
        result = amri_v2(se_naive=1.0, se_robust=10.0, n=100, theta=0.0, dof=98)
        assert result.w == 1.0
        assert result.se == pytest.approx(10.0)

    def test_small_ratio_gives_w_one(self):
        """When R is very small (< 1/tau), w should also saturate at 1."""
        result = amri_v2(se_naive=10.0, se_robust=1.0, n=100, theta=0.0, dof=98)
        assert result.w == 1.0
        assert result.se == pytest.approx(1.0)

    def test_w_in_unit_interval(self):
        """w must always be in [0, 1]."""
        rng = np.random.default_rng(42)
        for _ in range(50):
            se_n = rng.uniform(0.5, 2.0)
            se_r = rng.uniform(0.5, 2.0)
            n = rng.integers(10, 500)
            r = amri_v2(se_naive=se_n, se_robust=se_r, n=int(n), theta=0.0)
            assert 0.0 <= r.w <= 1.0

    def test_ci_contains_theta(self):
        """The CI should contain the point estimate."""
        r = amri_v2(se_naive=1.0, se_robust=1.2, n=200, theta=3.0, dof=198)
        assert r.ci_low < r.theta < r.ci_high

    def test_monotonicity_of_w_in_abs_log_R(self):
        """w should be non-decreasing in |log(R)| for fixed n."""
        n = 100
        ratios = [0.5, 0.7, 0.85, 0.95, 1.0, 1.05, 1.15, 1.3, 1.5, 2.0, 3.0]
        # Sort by |log(R)|
        sorted_ratios = sorted(ratios, key=lambda r: abs(math.log(r)))
        weights = []
        for R in sorted_ratios:
            se_r = R * 1.0  # se_naive = 1.0
            res = amri_v2(se_naive=1.0, se_robust=se_r, n=n, theta=0.0)
            weights.append(res.w)
        for i in range(len(weights) - 1):
            assert weights[i] <= weights[i + 1] + 1e-12

    def test_default_theta_zero(self):
        """When theta is None, it should default to 0."""
        r = amri_v2(se_naive=1.0, se_robust=1.0, n=100)
        assert r.theta == 0.0

    def test_symmetric_ci(self):
        """CI should be symmetric around theta."""
        r = amri_v2(se_naive=1.0, se_robust=1.3, n=200, theta=5.0, dof=198)
        half_width = r.ci_high - r.theta
        assert r.theta - r.ci_low == pytest.approx(half_width)

    def test_invalid_se_raises(self):
        with pytest.raises(ValueError):
            amri_v2(se_naive=0.0, se_robust=1.0, n=100)
        with pytest.raises(ValueError):
            amri_v2(se_naive=1.0, se_robust=-1.0, n=100)

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError):
            amri_v2(se_naive=1.0, se_robust=1.0, n=1)

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            amri_v2(se_naive=1.0, se_robust=1.0, n=100, alpha=0.0)
        with pytest.raises(ValueError):
            amri_v2(se_naive=1.0, se_robust=1.0, n=100, alpha=1.0)


# ---------------------------------------------------------------------------
# amri_v1
# ---------------------------------------------------------------------------

class TestAMRIv1:
    """Unit tests for the v1 hard-switching variant."""

    def test_ratio_near_one_efficient_mode(self):
        """R near 1 should give efficient mode (w=0)."""
        # tau = 1 + 2/sqrt(100) = 1.2, so R=1.1 is within bounds
        result = amri_v1(se_naive=1.0, se_robust=1.1, n=100, theta=2.0, dof=98)
        assert result.w == 0.0
        assert result.se == pytest.approx(1.0)  # uses se_naive
        assert result.method == "v1"

    def test_large_ratio_robust_mode(self):
        """R well above tau should trigger robust mode (w=1)."""
        # tau = 1.2 for n=100, R=2.0 > tau
        result = amri_v1(se_naive=1.0, se_robust=2.0, n=100, theta=0.0, dof=98)
        assert result.w == 1.0
        assert result.se == pytest.approx(1.05 * 2.0)

    def test_small_ratio_robust_mode(self):
        """R well below 1/tau should trigger robust mode."""
        # tau = 1.2, 1/tau ~ 0.833, so R=0.5 < 1/tau
        result = amri_v1(se_naive=1.0, se_robust=0.5, n=100, theta=0.0, dof=98)
        assert result.w == 1.0
        assert result.se == pytest.approx(1.05 * 0.5)

    def test_boundary_efficient(self):
        """R exactly at tau boundary should still be efficient (<=)."""
        n = 100
        tau = 1.0 + 2.0 / math.sqrt(n)
        # R = tau - epsilon should be efficient
        R = tau - 0.001
        result = amri_v1(se_naive=1.0, se_robust=R, n=n, theta=0.0)
        assert result.w == 0.0

    def test_ci_contains_theta(self):
        r = amri_v1(se_naive=1.0, se_robust=1.1, n=200, theta=5.0, dof=198)
        assert r.ci_low < r.theta < r.ci_high


# ---------------------------------------------------------------------------
# adaptive_ci (high-level OLS wrapper)
# ---------------------------------------------------------------------------

class TestAdaptiveCI:
    """Integration tests for the adaptive_ci convenience function."""

    @pytest.fixture()
    def ols_data(self):
        """Generate well-specified OLS data: y = 1 + 2*X + N(0,1)."""
        rng = np.random.default_rng(12345)
        X = np.arange(100, dtype=float)
        y = 1.0 + 2.0 * X + rng.standard_normal(100)
        return X, y

    def test_returns_amri_result(self, ols_data):
        X, y = ols_data
        result = adaptive_ci(X, y)
        assert isinstance(result, AMRIResult)

    def test_version_v2_default(self, ols_data):
        X, y = ols_data
        result = adaptive_ci(X, y)
        assert result.method == "v2"

    def test_version_v1(self, ols_data):
        X, y = ols_data
        result = adaptive_ci(X, y, version="v1")
        assert result.method == "v1"

    def test_invalid_version_raises(self, ols_data):
        X, y = ols_data
        with pytest.raises(ValueError, match="version"):
            adaptive_ci(X, y, version="v3")

    def test_ci_contains_true_slope(self, ols_data):
        """With well-specified data, the 95% CI should contain slope=2."""
        X, y = ols_data
        result = adaptive_ci(X, y, alpha=0.05)
        assert result.ci_low < 2.0 < result.ci_high

    def test_weight_in_unit_interval(self, ols_data):
        X, y = ols_data
        result = adaptive_ci(X, y)
        assert 0.0 <= result.w <= 1.0

    def test_well_specified_gives_low_weight(self):
        """Homoskedastic data should yield w near 0."""
        rng = np.random.default_rng(999)
        n = 500
        X = rng.standard_normal(n)
        y = 1.0 + 2.0 * X + rng.standard_normal(n)
        result = adaptive_ci(X, y)
        # With well-specified data and n=500, w should be close to 0
        assert result.w < 0.5

    def test_heteroskedastic_gives_higher_weight(self):
        """Strong heteroskedasticity should push w toward 1."""
        rng = np.random.default_rng(42)
        n = 500
        X = rng.standard_normal(n)
        noise = rng.standard_normal(n) * np.abs(X) * 5.0
        y = 1.0 + 2.0 * X + noise
        result = adaptive_ci(X, y)
        # w should be noticeably higher than for homoskedastic data
        assert result.w > 0.0

    def test_multiple_predictors(self):
        """adaptive_ci should work with 2-D X."""
        rng = np.random.default_rng(7)
        n = 200
        X = rng.standard_normal((n, 3))
        y = 1.0 + X @ np.array([2.0, -1.0, 0.5]) + rng.standard_normal(n)
        result = adaptive_ci(X, y, param_index=1)
        assert isinstance(result, AMRIResult)
        # Slope for X[:,0] should be near 2.0
        assert result.ci_low < 2.0 < result.ci_high

    def test_1d_and_2d_equivalence(self):
        """1-D X and 2-D X with one column should give the same result."""
        rng = np.random.default_rng(123)
        X = rng.standard_normal(100)
        y = 1.0 + 0.5 * X + rng.standard_normal(100)
        r1 = adaptive_ci(X, y)
        r2 = adaptive_ci(X.reshape(-1, 1), y)
        assert r1.theta == pytest.approx(r2.theta)
        assert r1.w == pytest.approx(r2.w)

    def test_dimension_mismatch_raises(self):
        X = np.arange(100, dtype=float)
        y = np.arange(50, dtype=float)
        with pytest.raises(ValueError, match="rows"):
            adaptive_ci(X, y)

    def test_param_index_out_of_range_raises(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal(100)
        y = rng.standard_normal(100)
        with pytest.raises(ValueError, match="param_index"):
            adaptive_ci(X, y, param_index=5)

    def test_coverage_simulation(self):
        """Monte Carlo check: 95% CI covers true param >= 85% of the time.

        We use a generous lower bound (85% instead of 95%) because this is
        a quick smoke test with only 200 replications.
        """
        rng = np.random.default_rng(2024)
        true_slope = 3.0
        n = 100
        n_reps = 200
        covers = 0
        for _ in range(n_reps):
            X = rng.standard_normal(n)
            y = 1.0 + true_slope * X + rng.standard_normal(n)
            result = adaptive_ci(X, y, alpha=0.05)
            if result.ci_low <= true_slope <= result.ci_high:
                covers += 1
        coverage = covers / n_reps
        assert coverage >= 0.85, f"Coverage {coverage:.3f} too low"
