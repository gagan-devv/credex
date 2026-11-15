"""
Models module for Credex.

Exposes:
- Baseline algorithms replicating Li & Wu (2024) and probability calibration
- Analytical cost-sensitive objective functions and derivatives
- Cost-sensitive LightGBM, XGBoost, and CatBoost classifier wrappers
- Economic loss threshold optimizer for unit misclassification and portfolio dollar losses
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
from src.models.cost_sensitive import (
    CostSensitiveCatBoostClassifier,
    CostSensitiveCatBoostObjective,
    CostSensitiveLGBMClassifier,
    CostSensitiveXGBClassifier,
    make_lightgbm_eval_metric,
    make_lightgbm_objective,
    make_xgboost_eval_metric,
    make_xgboost_objective,
)
from src.models.objective_functions import (
    cost_sensitive_gradient,
    cost_sensitive_gradients,
    cost_sensitive_hessian,
    cost_sensitive_loss,
    cost_sensitive_loss_elementwise,
    numerical_gradient,
    numerical_hessian,
    sigmoid,
)
from src.models.threshold_optimizer import (
    ThresholdOptimizationResult,
    ThresholdOptimizer,
    compute_confusion_at_thresholds,
    compute_portfolio_dollar_loss,
    compute_unit_expected_loss,
    parse_term_in_months,
)

__all__ = [
    # Baseline Models
    "BASELINE_MODEL_NAMES",
    "DEFAULT_BASELINE_PARAMS",
    "PUBLISHED_BENCHMARKS",
    "BaselineModel",
    "get_baseline_model",
    "get_all_baseline_models",
    "run_baseline_replication",
    # Probability Calibration
    "ProbabilityCalibrator",
    "OutOfFoldCalibrator",
    "compute_brier_score",
    "compute_expected_calibration_error",
    "compute_calibration_curve",
    "evaluate_calibration",
    # Objective Functions & Gradients
    "sigmoid",
    "cost_sensitive_loss",
    "cost_sensitive_loss_elementwise",
    "cost_sensitive_gradients",
    "cost_sensitive_gradient",
    "cost_sensitive_hessian",
    "numerical_gradient",
    "numerical_hessian",
    # Cost-Sensitive Tree Models & Wrappers
    "make_lightgbm_objective",
    "make_lightgbm_eval_metric",
    "make_xgboost_objective",
    "make_xgboost_eval_metric",
    "CostSensitiveCatBoostObjective",
    "CostSensitiveLGBMClassifier",
    "CostSensitiveXGBClassifier",
    "CostSensitiveCatBoostClassifier",
    # Economic Threshold Optimization
    "ThresholdOptimizer",
    "ThresholdOptimizationResult",
    "parse_term_in_months",
    "compute_confusion_at_thresholds",
    "compute_unit_expected_loss",
    "compute_portfolio_dollar_loss",
]
