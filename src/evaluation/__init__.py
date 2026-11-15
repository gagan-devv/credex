"""
Evaluation and benchmark metrics suite for Credex.
"""

from src.evaluation.metrics import (
    CreditEvaluationResult,
    compute_brier_score,
    compute_dollar_savings,
    compute_expected_unit_loss,
    compute_portfolio_loss,
    compute_statistical_metrics,
    evaluate_credit_model,
    evaluate_lgd_sensitivity,
)
from src.evaluation.ablation_runner import (
    FACTORIAL_CUTOFF_OPTIONS,
    FACTORIAL_OBJECTIVE_OPTIONS,
    FACTORIAL_SAMPLING_OPTIONS,
    FactorialAblationRunner,
    instantiate_ablation_model,
    run_factorial_ablation,
)

__all__ = [
    "CreditEvaluationResult",
    "compute_brier_score",
    "compute_dollar_savings",
    "compute_expected_unit_loss",
    "compute_portfolio_loss",
    "compute_statistical_metrics",
    "evaluate_credit_model",
    "evaluate_lgd_sensitivity",
    "FACTORIAL_CUTOFF_OPTIONS",
    "FACTORIAL_OBJECTIVE_OPTIONS",
    "FACTORIAL_SAMPLING_OPTIONS",
    "FactorialAblationRunner",
    "instantiate_ablation_model",
    "run_factorial_ablation",
]
