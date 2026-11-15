"""
Evaluation and benchmark metrics suite for Credex (CREDEX-13..15, CREDEX-18).
"""

from src.evaluation.ablation_runner import (
    FACTORIAL_CUTOFF_OPTIONS,
    FACTORIAL_OBJECTIVE_OPTIONS,
    FACTORIAL_SAMPLING_OPTIONS,
    FactorialAblationRunner,
    instantiate_ablation_model,
    run_factorial_ablation,
)
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
from src.evaluation.paper_artifacts import (
    PaperArtifactGenerator,
    PaperArtifactsResult,
    generate_all_paper_artifacts,
    generate_fig1_roc_pr_comparison,
    generate_fig2_expected_loss_curves,
    generate_fig3_shap_global_beeswarm,
    generate_fig4_actionable_recourse_waterfall,
    generate_paper_results_digest,
    generate_table1_descriptive_stats,
    generate_table2_li_wu_replication,
    generate_table3_cost_sensitive_ablation,
    generate_table4_economic_dollar_savings,
    validate_latex_table,
)

__all__ = [
    # Metrics
    "CreditEvaluationResult",
    "compute_brier_score",
    "compute_dollar_savings",
    "compute_expected_unit_loss",
    "compute_portfolio_loss",
    "compute_statistical_metrics",
    "evaluate_credit_model",
    "evaluate_lgd_sensitivity",
    # Factorial Ablation
    "FACTORIAL_CUTOFF_OPTIONS",
    "FACTORIAL_OBJECTIVE_OPTIONS",
    "FACTORIAL_SAMPLING_OPTIONS",
    "FactorialAblationRunner",
    "instantiate_ablation_model",
    "run_factorial_ablation",
    # Paper Artifacts (CREDEX-18)
    "PaperArtifactGenerator",
    "PaperArtifactsResult",
    "generate_all_paper_artifacts",
    "generate_table1_descriptive_stats",
    "generate_table2_li_wu_replication",
    "generate_table3_cost_sensitive_ablation",
    "generate_table4_economic_dollar_savings",
    "generate_fig1_roc_pr_comparison",
    "generate_fig2_expected_loss_curves",
    "generate_fig3_shap_global_beeswarm",
    "generate_fig4_actionable_recourse_waterfall",
    "generate_paper_results_digest",
    "validate_latex_table",
]
