"""
Models module for Credex: Baseline algorithms, calibration, cost-sensitive estimators,
loss objectives, and threshold optimization.
"""

from src.models.baselines import (
    BASELINE_MODEL_NAMES,
    DEFAULT_BASELINE_PARAMS,
    PUBLISHED_BENCHMARKS,
    BaselineModel,
    get_all_baseline_models,
    get_baseline_model,
    run_baseline_replication,
)
from src.models.calibration import (
    OutOfFoldCalibrator,
    ProbabilityCalibrator,
    compute_brier_score,
    compute_calibration_curve,
    compute_expected_calibration_error,
    evaluate_calibration,
)

__all__ = [
    "BASELINE_MODEL_NAMES",
    "DEFAULT_BASELINE_PARAMS",
    "PUBLISHED_BENCHMARKS",
    "BaselineModel",
    "get_baseline_model",
    "get_all_baseline_models",
    "run_baseline_replication",
    "ProbabilityCalibrator",
    "OutOfFoldCalibrator",
    "compute_brier_score",
    "compute_expected_calibration_error",
    "compute_calibration_curve",
    "evaluate_calibration",
]
