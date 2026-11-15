"""
Data processing, ingestion, feature engineering, and partitioning modules for Credex.
"""

from src.data.loader import (
    COMPLETED_LOAN_STATUSES,
    TARGET_MAPPING,
    clear_loader_cache,
    filter_completed_loans,
    generate_synthetic_german_credit,
    generate_synthetic_lending_club,
    load_german_credit,
    load_lending_club,
)
from src.data.feature_engineering import (
    BASELINE_CATEGORICAL,
    BASELINE_NUMERICAL,
    ENGINEERED_RATIOS,
    PRICING_FEATURES,
    CreditFeatureEngineer,
    calculate_delinquency_velocity,
    calculate_liquidity_buffer,
    calculate_loan_to_income,
    calculate_payment_to_income,
    engineer_solvency_ratios,
    get_feature_matrix,
    get_feature_schema,
)
from src.data.preprocessor import (
    DEFAULT_WINSORIZE_FEATURES,
    CreditDataPreprocessor,
    fit_transform_out_of_fold,
)
from src.data.partitioning import (
    LEAKAGE_COLUMNS,
    audit_feature_matrix_leakage,
    create_oot_split,
    create_stratified_kfold_splits,
    strip_leakage_columns,
)

__all__ = [
    # Loader
    "COMPLETED_LOAN_STATUSES",
    "TARGET_MAPPING",
    "clear_loader_cache",
    "filter_completed_loans",
    "generate_synthetic_german_credit",
    "generate_synthetic_lending_club",
    "load_german_credit",
    "load_lending_club",
    # Feature Engineering
    "BASELINE_CATEGORICAL",
    "BASELINE_NUMERICAL",
    "ENGINEERED_RATIOS",
    "PRICING_FEATURES",
    "CreditFeatureEngineer",
    "calculate_delinquency_velocity",
    "calculate_liquidity_buffer",
    "calculate_loan_to_income",
    "calculate_payment_to_income",
    "engineer_solvency_ratios",
    "get_feature_matrix",
    "get_feature_schema",
    # Preprocessor
    "DEFAULT_WINSORIZE_FEATURES",
    "CreditDataPreprocessor",
    "fit_transform_out_of_fold",
    # Partitioning & Leakage
    "LEAKAGE_COLUMNS",
    "audit_feature_matrix_leakage",
    "create_oot_split",
    "create_stratified_kfold_splits",
    "strip_leakage_columns",
]
