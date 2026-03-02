"""
Tests for amri.diagnostics — se_ratio, blending_weight, and amri_diagnose.
"""

import math

import numpy as np
import pytest

from amri.diagnostics import amri_diagnose, blending_weight, se_ratio


# ---------------------------------------------------------------------------
# se_ratio
# ---------------------------------------------------------------------------

class TestSERatio:
    """Unit tests for the se_ratio function."""

    @pytest.fixture()
    def homoskedastic_data(self):
        rng = np.random.default_rng(42)
        n = 300
        X = rng.standard_normal(n)
        y = 1.0 + 2.0 * X + rng.standard_normal(n)
        return X, y

    def test_returns_positive_float(self, homoskedastic_data):
        X, y = homoskedastic_data
        R = se_ratio(X, y)
        assert isinstance(R, float)
        assert R > 0.0

    def test_near_one_for_homoskedastic(self, homoskedastic_data):
        """Under homoskedasticity, HC3/naive should be near 1."""
        X, y = homoskedastic_data
        R = se_ratio(X, y)
        assert 0.7 < R < 1.5, f"Expected R near 1, got {R}"

    def test_larger_for_heteroskedastic(self):
        """Heteroskedastic data should push the ratio away from 1."""
        rng = np.random.default_rng(42)
        n = 500
        X = rng.standard_normal(n)
        noise = rng.standard_normal(n) * (1.0 + 3.0 * np.abs(X))
        y = 1.0 + 2.0 * X + noise
        R = se_ratio(X, y)
        # With strong heteroskedasticity, R should deviate from 1
        assert abs(R - 1.0) > 0.01

    def test_invalid_param_index_raises(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal(100)
        y = rng.standard_normal(100)
        with pytest.raises(ValueError, match="param_index"):
            se_ratio(X, y, param_index=5)


# ---------------------------------------------------------------------------
# blending_weight
# ---------------------------------------------------------------------------

class TestBlendingWeight:
    """Unit tests for the blending_weight function."""

    def test_ratio_one_gives_zero(self):
        """R=1 => |log(1)|=0 => w=0 (assuming c1>0 or c1/sqrt(n) > 0)."""
        w = blending_weight(R=1.0, n=100)
        assert w == 0.0

    def test_large_ratio_gives_one(self):
        w = blending_weight(R=100.0, n=100)
        assert w == 1.0

    def test_small_ratio_gives_one(self):
        w = blending_weight(R=0.01, n=100)
        assert w == 1.0

    def test_bounds_zero_one(self):
        """w must always be in [0, 1] for any positive R and valid n."""
        rng = np.random.default_rng(123)
        for _ in range(100):
            R = rng.lognormal(0, 1)
            n = rng.integers(5, 1000)
            w = blending_weight(R, int(n))
            assert 0.0 <= w <= 1.0

    def test_monotonic_in_abs_log_R(self):
        """w should be non-decreasing as |log(R)| increases."""
        n = 100
        ratios = np.exp(np.linspace(0, 3, 50))
        weights = [blending_weight(float(R), n) for R in ratios]
        for i in range(len(weights) - 1):
            assert weights[i] <= weights[i + 1] + 1e-12

    def test_symmetric_in_log_R(self):
        """w(R) should equal w(1/R)."""
        for R in [0.5, 0.8, 1.2, 2.0, 3.0]:
            w1 = blending_weight(R, n=100)
            w2 = blending_weight(1.0 / R, n=100)
            assert w1 == pytest.approx(w2)

    def test_invalid_R_raises(self):
        with pytest.raises(ValueError):
            blending_weight(R=0.0, n=100)
        with pytest.raises(ValueError):
            blending_weight(R=-1.0, n=100)

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError):
            blending_weight(R=1.0, n=0)

    def test_c1_c2_ordering(self):
        with pytest.raises(ValueError):
            blending_weight(R=1.5, n=100, c1=2.0, c2=1.0)

    def test_custom_c1_c2(self):
        """With custom constants, the transition region changes."""
        # Tight region: c1=0.5, c2=0.6 => very narrow soft region
        w_tight = blending_weight(R=1.5, n=100, c1=0.5, c2=0.6)
        # Wide region: c1=0.5, c2=5.0 => very wide soft region
        w_wide = blending_weight(R=1.5, n=100, c1=0.5, c2=5.0)
        # For the same R, the tight window should give higher w
        assert w_tight >= w_wide


# ---------------------------------------------------------------------------
# amri_diagnose
# ---------------------------------------------------------------------------

class TestAMRIDiagnose:
    """Unit tests for the amri_diagnose function."""

    @pytest.fixture()
    def homoskedastic_data(self):
        rng = np.random.default_rng(42)
        n = 500
        X = rng.standard_normal(n)
        y = 1.0 + 2.0 * X + rng.standard_normal(n)
        return X, y

    @pytest.fixture()
    def heteroskedastic_data(self):
        rng = np.random.default_rng(42)
        n = 500
        X = rng.standard_normal(n)
        noise = rng.standard_normal(n) * (1.0 + 5.0 * np.abs(X))
        y = 1.0 + 2.0 * X + noise
        return X, y

    def test_returns_expected_keys(self, homoskedastic_data):
        X, y = homoskedastic_data
        diag = amri_diagnose(X, y)
        expected = {"ratio", "w", "se_naive", "se_hc3", "mode", "n", "recommendation"}
        assert set(diag.keys()) == expected

    def test_ratio_positive(self, homoskedastic_data):
        X, y = homoskedastic_data
        diag = amri_diagnose(X, y)
        assert diag["ratio"] > 0

    def test_w_in_unit_interval(self, homoskedastic_data):
        X, y = homoskedastic_data
        diag = amri_diagnose(X, y)
        assert 0.0 <= diag["w"] <= 1.0

    def test_se_positive(self, homoskedastic_data):
        X, y = homoskedastic_data
        diag = amri_diagnose(X, y)
        assert diag["se_naive"] > 0
        assert diag["se_hc3"] > 0

    def test_n_correct(self, homoskedastic_data):
        X, y = homoskedastic_data
        diag = amri_diagnose(X, y)
        assert diag["n"] == 500

    def test_mode_is_valid(self, homoskedastic_data):
        X, y = homoskedastic_data
        diag = amri_diagnose(X, y)
        assert diag["mode"] in ("efficient", "blending", "robust")

    def test_efficient_mode_for_homoskedastic(self, homoskedastic_data):
        """Well-specified model should tend toward efficient mode."""
        X, y = homoskedastic_data
        diag = amri_diagnose(X, y)
        # With n=500 and homoskedastic noise, expect efficient or at most
        # light blending
        assert diag["mode"] in ("efficient", "blending")

    def test_robust_mode_for_heteroskedastic(self, heteroskedastic_data):
        """Strongly heteroskedastic data should push toward robust mode."""
        X, y = heteroskedastic_data
        diag = amri_diagnose(X, y)
        # With strong heteroskedasticity, expect blending or robust
        assert diag["mode"] in ("blending", "robust")

    def test_recommendation_is_string(self, homoskedastic_data):
        X, y = homoskedastic_data
        diag = amri_diagnose(X, y)
        assert isinstance(diag["recommendation"], str)
        assert len(diag["recommendation"]) > 10

    def test_recommendation_mentions_ratio(self, homoskedastic_data):
        X, y = homoskedastic_data
        diag = amri_diagnose(X, y)
        # The recommendation should mention the ratio value
        assert "R =" in diag["recommendation"] or "ratio" in diag["recommendation"].lower()
