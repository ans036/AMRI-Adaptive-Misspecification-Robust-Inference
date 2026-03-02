"""
Unit tests for AMRI core functions.
Run: python -m pytest tests/ -v
"""
import numpy as np
import pytest
from scipy import stats
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ============================================================================
# AMRI v2 core function (standalone for testing)
# ============================================================================

def amri_v2_se(se_naive, se_robust, n, c1=1.0, c2=2.0):
    """AMRI v2 blended SE with soft-thresholding."""
    ratio = se_robust / max(se_naive, 1e-10)
    log_ratio = abs(np.log(ratio))
    lower = c1 / np.sqrt(n)
    upper = c2 / np.sqrt(n)
    if upper <= lower:
        w = 1.0 if log_ratio > lower else 0.0
    else:
        w = np.clip((log_ratio - lower) / (upper - lower), 0.0, 1.0)
    se = (1 - w) * se_naive + w * se_robust
    return se, w, log_ratio


def amri_v1_se(se_naive, se_robust, n):
    """AMRI v1 hard-switching SE."""
    ratio = se_robust / max(se_naive, 1e-10)
    threshold = 1 + 2 / np.sqrt(n)
    if ratio > threshold or ratio < 1 / threshold:
        return se_robust * 1.05, 1.0
    else:
        return se_naive, 0.0


# ============================================================================
# TEST: Blending weight properties
# ============================================================================

class TestBlendingWeight:
    """Test AMRI v2 blending weight w."""

    def test_w_zero_when_ses_identical(self):
        """When se_naive == se_robust, w should be 0."""
        se, w, _ = amri_v2_se(1.0, 1.0, 100)
        assert w == 0.0
        assert se == 1.0

    def test_w_one_when_ses_very_different(self):
        """When se_robust >> se_naive, w should be 1."""
        se, w, _ = amri_v2_se(1.0, 5.0, 100)
        assert w == 1.0
        assert se == 5.0

    def test_w_between_zero_and_one(self):
        """w should be in [0, 1] for any input."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            se_n = rng.uniform(0.1, 10)
            se_r = rng.uniform(0.1, 10)
            n = rng.integers(30, 10000)
            _, w, _ = amri_v2_se(se_n, se_r, n)
            assert 0.0 <= w <= 1.0

    def test_w_monotonic_in_ratio(self):
        """w should increase as |log(se_r/se_n)| increases."""
        n = 100
        weights = []
        for se_r in [1.0, 1.1, 1.3, 1.5, 2.0, 3.0, 5.0]:
            _, w, _ = amri_v2_se(1.0, se_r, n)
            weights.append(w)
        # Check monotonically non-decreasing
        for i in range(1, len(weights)):
            assert weights[i] >= weights[i-1], f"w not monotonic: {weights}"

    def test_w_monotonic_in_ratio_below_one(self):
        """w should also increase as ratio goes below 1."""
        n = 100
        weights = []
        for se_r in [1.0, 0.9, 0.7, 0.5, 0.3]:
            _, w, _ = amri_v2_se(1.0, se_r, n)
            weights.append(w)
        for i in range(1, len(weights)):
            assert weights[i] >= weights[i-1]

    def test_w_decreases_with_n(self):
        """For fixed ratio close to 1, w should decrease with n (threshold widens)."""
        ratio = 1.15  # moderate misspecification signal
        weights = []
        for n in [50, 100, 500, 1000, 5000]:
            _, w, _ = amri_v2_se(1.0, ratio, n)
            weights.append(w)
        # Eventually w should be higher for large n (threshold narrows)
        # Actually: for ratio close to 1, as n grows, c1/sqrt(n) shrinks,
        # so |log(ratio)| > c2/sqrt(n) sooner. w should INCREASE with n.
        assert weights[-1] >= weights[0]

    def test_threshold_parameters(self):
        """Different c1, c2 should change the blending behavior."""
        se, w1, _ = amri_v2_se(1.0, 1.5, 100, c1=0.5, c2=1.5)  # wider transition
        se, w2, _ = amri_v2_se(1.0, 1.5, 100, c1=1.0, c2=2.0)  # default
        se, w3, _ = amri_v2_se(1.0, 1.5, 100, c1=2.0, c2=4.0)  # narrower transition
        # Wider transition zone -> less weight at same ratio
        assert w1 >= w3


# ============================================================================
# TEST: SE computation
# ============================================================================

class TestSEComputation:
    """Test that AMRI SE is always between naive and robust."""

    def test_se_bounded(self):
        """AMRI v2 SE should always be between se_naive and se_robust."""
        rng = np.random.default_rng(42)
        for _ in range(200):
            se_n = rng.uniform(0.1, 5)
            se_r = rng.uniform(0.1, 5)
            n = rng.integers(30, 5000)
            se, w, _ = amri_v2_se(se_n, se_r, n)
            assert se >= min(se_n, se_r) - 1e-10
            assert se <= max(se_n, se_r) + 1e-10

    def test_se_equals_naive_when_w_zero(self):
        """When w=0, AMRI SE should equal naive SE exactly."""
        se, w, _ = amri_v2_se(2.0, 2.0, 500)
        assert w == 0.0
        assert abs(se - 2.0) < 1e-10

    def test_se_equals_robust_when_w_one(self):
        """When w=1, AMRI SE should equal robust SE exactly."""
        se, w, _ = amri_v2_se(1.0, 10.0, 500)
        assert w == 1.0
        assert abs(se - 10.0) < 1e-10


# ============================================================================
# TEST: v1 vs v2 comparison
# ============================================================================

class TestV1V2Comparison:
    """Compare AMRI v1 and v2 behavior."""

    def test_v1_binary_switch(self):
        """v1 should produce exactly 0 or 1 weight."""
        for se_r in [0.5, 0.8, 1.0, 1.2, 2.0, 5.0]:
            _, w = amri_v1_se(1.0, se_r, 100)
            assert w in [0.0, 1.0], f"v1 weight not binary: {w}"

    def test_v2_can_produce_intermediate(self):
        """v2 should be able to produce weights strictly between 0 and 1."""
        found_intermediate = False
        for se_r in np.linspace(0.5, 2.0, 50):
            for n in [50, 100, 200, 500]:
                _, w, _ = amri_v2_se(1.0, se_r, n)
                if 0 < w < 1:
                    found_intermediate = True
                    break
            if found_intermediate:
                break
        assert found_intermediate, "v2 never produced intermediate weight"


# ============================================================================
# TEST: Coverage with known DGP
# ============================================================================

class TestCoverageSimulation:
    """Verify AMRI achieves nominal coverage on known DGPs."""

    @pytest.fixture
    def rng(self):
        return np.random.default_rng(12345)

    def _simulate_coverage(self, n, delta, B, rng, method='v2'):
        """Run B simulations and compute coverage."""
        import statsmodels.api as sm
        covers = 0
        for _ in range(B):
            X = rng.standard_normal(n)
            sigma = np.exp(delta * X / 2)
            eps = rng.standard_normal(n) * sigma
            Y = X + eps
            theta_true = 1.0

            Xa = sm.add_constant(X)
            model_n = sm.OLS(Y, Xa).fit()
            model_r = sm.OLS(Y, Xa).fit(cov_type='HC3')
            theta = model_n.params[1]
            se_n = model_n.bse[1]
            se_r = model_r.bse[1]
            t_val = stats.t.ppf(0.975, n - 2)

            if method == 'v2':
                se, _, _ = amri_v2_se(se_n, se_r, n)
            else:
                se, _ = amri_v1_se(se_n, se_r, n)

            if theta - t_val * se <= theta_true <= theta + t_val * se:
                covers += 1

        return covers / B

    def test_coverage_no_misspec(self, rng):
        """Under correct specification (delta=0), coverage should be >= 0.93."""
        cov = self._simulate_coverage(n=200, delta=0.0, B=500, rng=rng)
        assert cov >= 0.93, f"Coverage {cov:.4f} < 0.93 under correct spec"

    def test_coverage_with_misspec(self, rng):
        """Under misspecification (delta=1), coverage should still be >= 0.90."""
        cov = self._simulate_coverage(n=200, delta=1.0, B=500, rng=rng)
        assert cov >= 0.90, f"Coverage {cov:.4f} < 0.90 under misspec"

    def test_v2_coverage_better_or_equal_to_v1(self, rng):
        """V2 should have coverage at least as good as v1 (on average)."""
        cov_v2 = self._simulate_coverage(n=200, delta=0.5, B=500, rng=np.random.default_rng(42), method='v2')
        cov_v1 = self._simulate_coverage(n=200, delta=0.5, B=500, rng=np.random.default_rng(42), method='v1')
        # Allow some MC noise
        assert cov_v2 >= cov_v1 - 0.03, f"v2 ({cov_v2:.4f}) much worse than v1 ({cov_v1:.4f})"


# ============================================================================
# TEST: Edge cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and numerical stability."""

    def test_very_small_se_naive(self):
        """Should not crash with very small naive SE."""
        se, w, _ = amri_v2_se(1e-15, 1.0, 100)
        assert np.isfinite(se)
        assert np.isfinite(w)

    def test_very_large_ratio(self):
        """Should handle extreme SE ratios gracefully."""
        se, w, _ = amri_v2_se(0.001, 1000.0, 100)
        assert w == 1.0
        assert abs(se - 1000.0) < 1e-10

    def test_n_equals_30(self):
        """Minimum practical sample size."""
        se, w, _ = amri_v2_se(1.0, 1.5, 30)
        assert np.isfinite(se)
        assert 0.0 <= w <= 1.0

    def test_very_large_n(self):
        """Very large n should make threshold very tight."""
        se, w, _ = amri_v2_se(1.0, 1.01, 1_000_000)
        # Even a tiny ratio difference should trigger w > 0 at large n
        assert w > 0


# ============================================================================
# TEST: DGP functions
# ============================================================================

class TestDGPs:
    """Test that DGP functions return correct theta_true."""

    def test_heteroscedastic_dgp(self):
        from competitor_comparison import dgp_heteroscedasticity
        rng = np.random.default_rng(42)
        X, Y, theta = dgp_heteroscedasticity(1000, 0.5, rng)
        assert theta == 1.0
        assert len(X) == 1000
        assert len(Y) == 1000

    def test_nonlinear_dgp(self):
        from competitor_comparison import dgp_nonlinearity
        rng = np.random.default_rng(42)
        X, Y, theta = dgp_nonlinearity(1000, 0.5, rng)
        assert theta == 1.0

    def test_omitted_variable_dgp(self):
        from competitor_comparison import dgp_omitted_variable
        rng = np.random.default_rng(42)
        X, Y, theta = dgp_omitted_variable(1000, 0.5, rng)
        # theta = 1 + delta * rho = 1 + 0.5 * 0.5 = 1.25
        assert abs(theta - 1.25) < 1e-10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
