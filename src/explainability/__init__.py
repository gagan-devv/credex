"""
Explainability & Actionable Recourse Engine for Credex (CREDEX-16 & CREDEX-17).

Exposes:
- TreeSHAP attribution and additivity verification engine (`TreeSHAPExplainer`, `TreeSHAPResult`)
- High-DPI publication vector visualizer (`plot_beeswarm`, `plot_feature_dependence`, `plot_waterfall`, `plot_recourse_waterfall_comparison`, `generate_publication_figures`)
- Constrained counterfactual recourse & ECOA Regulation B adverse action engine (`ActionableRecourseEngine`, `ActionableRecourseResult`, `RecourseOption`)
"""

from src.explainability.adverse_action import (
    DEFAULT_IMMUTABLE_FEATURES,
    DEFAULT_MUTABLE_FEATURES,
    FEATURE_LABELS,
    ActionableRecourseEngine,
    ActionableRecourseResult,
    RecourseOption,
)
from src.explainability.shap_explainer import (
    TreeSHAPExplainer,
    TreeSHAPResult,
    extract_tree_estimator,
    get_native_tree_output,
)
from src.explainability.visualizer import (
    generate_publication_figures,
    plot_beeswarm,
    plot_beeswarm_summary,
    plot_feature_dependence,
    plot_recourse_waterfall_comparison,
    plot_adverse_action_waterfall,
    plot_waterfall,
)

__all__ = [
    # TreeSHAP Engine
    "TreeSHAPExplainer",
    "TreeSHAPResult",
    "extract_tree_estimator",
    "get_native_tree_output",
    # Publication Visualizer
    "plot_beeswarm",
    "plot_feature_dependence",
    "plot_waterfall",
    "plot_recourse_waterfall_comparison",
    "generate_publication_figures",
    # Actionable Recourse & Adverse Action Engine
    "ActionableRecourseEngine",
    "ActionableRecourseResult",
    "RecourseOption",
    "DEFAULT_MUTABLE_FEATURES",
    "DEFAULT_IMMUTABLE_FEATURES",
    "FEATURE_LABELS",
]
