"""
AMRI: Adaptive Misspecification-Robust Inference
=================================================

Adaptive confidence intervals that smoothly blend model-based and sandwich
standard errors using soft-threshold blending of the SE ratio diagnostic.

Quick start::

    from amri import adaptive_ci
    result = adaptive_ci(X, y)
    print(f"CI: [{result.ci_low:.3f}, {result.ci_high:.3f}]")
    print(f"Blending weight: {result.w:.3f}")

The blending weight ``w`` controls adaptation:

- ``w = 0``: Model is well-specified; use efficient (model-based) SE.
- ``0 < w < 1``: Soft blending between model-based and robust SE.
- ``w = 1``: Model is misspecified; use full robust (sandwich) SE.

Two versions are available:

- **v2** (default): Soft-threshold blending with near-minimax optimal constants.
- **v1**: Hard-switching variant with a fixed threshold.

Reference:
    "Adaptive Misspecification-Robust Confidence Intervals:
     Near-Minimax Optimal Inference via Soft-Threshold Blending"
"""

from amri.core import adaptive_ci, amri_v2, amri_v1, AMRIResult
from amri.estimators import (
    OLSEstimator,
    MultipleOLSEstimator,
    LogisticEstimator,
    PoissonEstimator,
)
from amri.diagnostics import se_ratio, blending_weight, amri_diagnose

__version__ = "0.1.0"
__all__ = [
    "adaptive_ci",
    "amri_v2",
    "amri_v1",
    "AMRIResult",
    "OLSEstimator",
    "MultipleOLSEstimator",
    "LogisticEstimator",
    "PoissonEstimator",
    "se_ratio",
    "blending_weight",
    "amri_diagnose",
]
