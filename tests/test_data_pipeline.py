"""
Comprehensive Unit Tests and Hypothesis Property-Based Tests for Data Pipeline.
Validates ingestion, out-of-fold preprocessing, financial winsorization,
solvency ratios, pricing feature ablation, and stratified partitioning.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.data.feature_engineering import (
    PRICING_FEATURES,
    calculate_delinquency_velocity,
    calculate_liquidity_buffer,
    calculate_loan_to_income,
    calculate_payment_to_income,
    engineer_solvency_ratios,
    get_feature_matrix,
    get_feature_schema,
)
from src.data.loader import (
    filter_completed_loans,
    generate_synthetic_german_credit,
    generate_synthetic_lending_club,
    load_german_credit,
    load_lending_club,
)
from src.data.partitioning import (
    create_oot_split,
    create_stratified_kfold_splits,
)
from src.data.preprocessor import (
    CreditDataPreprocessor,
    DEFAULT_WINSORIZE_FEATURES,
)


# =====================================================================
# Ingestion & Loader Tests
# =====================================================================

def test_filter_completed_loans():
    """Verify filter_completed_loans retains only terminal contracts and maps target."""
    raw_df = pd.DataFrame(
        {
            "loan_status": [
                "Fully Paid",
                "Charged Off",
                "Default",
                "Current",
                "In Grace Period",
                "Late (31-120 days)",
                "Fully Paid",
            ],
            "loan_amount": [10000, 15000, 20000, 5000, 12000, 8000, 7000],
        }
    )
    filtered = filter_completed_loans(raw_df)

    assert len(filtered) == 4
    assert set(filtered["loan_status"].unique()) == {0, 1}
    assert (filtered["loan_status"] == pd.Series([0, 1, 1, 0], index=filtered.index)).all()


def test_load_lending_club_sampling_and_stratification():
    """Verify dual-mode support: stratified sampling vs full dataset."""
    full_df = generate_synthetic_lending_club(n_samples=1000, random_state=42)
    pop_default_rate = full_df["loan_status"].mean()

    # Stratified sample mode (sample_size = 400)
    sample_df = load_lending_club(
        path=None,
        sample_size=400,
        use_full_dataset=False,
        random_state=42,
        fallback_to_synthetic=True,
        n_synthetic=1000,
    )
    assert len(sample_df) == 400
    sample_default_rate = sample_df["loan_status"].mean()

    # Default rate in sample must match population within +/- 0.5% (or 0.01 on small synthetic)
    assert abs(sample_default_rate - pop_default_rate) < 0.01


@settings(max_examples=15, deadline=None)
@given(
    sample_size=st.integers(min_value=200, max_value=600),
    seed=st.integers(min_value=1, max_value=5000),
)
def test_hypothesis_stratified_sample_distribution(sample_size, seed):
    """Property test: Target class distribution in sample matches population within 0.5%."""
    df = generate_synthetic_lending_club(n_samples=1000, random_state=seed)
    pop_rate = df["loan_status"].mean()

    from sklearn.model_selection import train_test_split
    sampled, _ = train_test_split(
        df,
        train_size=sample_size,
        stratify=df["loan_status"],
        random_state=seed,
    )
    sample_rate = sampled["loan_status"].mean()
    assert abs(sample_rate - pop_rate) <= 0.005


def test_load_german_credit():
    """Verify UCI South German Credit loader standardizes schema and target."""
    german_df = load_german_credit(fallback_to_synthetic=True, random_state=42)
    assert len(german_df) == 1000
    assert "loan_status" in german_df.columns
    assert set(german_df["loan_status"].unique()).issubset({0, 1})
    assert "duration" in german_df.columns
    assert "amount" in german_df.columns


# =====================================================================
# Solvency Ratios & Feature Engineering Tests
# =====================================================================

def test_solvency_ratios_exact_calculation():
    """Verify mathematical calculation of all 4 solvency ratios."""
    df = pd.DataFrame(
        {
            "installment": [500.0, 300.0],
            "annual_income": [60000.0, 36000.0],
            "loan_amount": [15000.0, 12000.0],
            "total_rev_hi_lim": [25000.0, 10000.0],
            "revol_bal": [5000.0, 8000.0],
            "delinq_2yrs": [1, 0],
            "total_acc": [10, 5],
        }
    )
    augmented = engineer_solvency_ratios(df)

    # PTI: installment / (annual_income / 12)
    # Row 0: 500 / (60000 / 12) = 500 / 5000 = 0.10
    # Row 1: 300 / (36000 / 12) = 300 / 3000 = 0.10
    np.testing.assert_allclose(augmented["payment_to_income"].values, [0.10, 0.10], rtol=1e-5)

    # LTI: loan_amount / annual_income
    # Row 0: 15000 / 60000 = 0.25
    # Row 1: 12000 / 36000 = 0.333333...
    np.testing.assert_allclose(augmented["loan_to_income"].values, [0.25, 12000.0 / 36000.0], rtol=1e-5)

    # Liquidity Buffer: total_rev_hi_lim - revol_bal
    # Row 0: 25000 - 5000 = 20000
    # Row 1: 10000 - 8000 = 2000
    np.testing.assert_allclose(augmented["liquidity_buffer"].values, [20000.0, 2000.0], rtol=1e-5)

    # Delinquency Velocity: delinq_2yrs / total_acc
    # Row 0: 1 / 10 = 0.10
    # Row 1: 0 / 5 = 0.0
    np.testing.assert_allclose(augmented["delinquency_velocity"].values, [0.10, 0.0], rtol=1e-5)


def test_solvency_ratios_division_by_zero_guards():
    """Verify solvency ratio calculations guard against zero and negative denominators."""
    df = pd.DataFrame(
        {
            "installment": [500.0, 200.0],
            "annual_income": [0.0, -5000.0],  # Zero and negative income
            "loan_amount": [10000.0, 5000.0],
            "delinq_2yrs": [2, 1],
            "total_acc": [0, -1],  # Zero and negative total accounts
        }
    )
    augmented = engineer_solvency_ratios(df)

    assert not np.isnan(augmented["payment_to_income"].values).any()
    assert not np.isinf(augmented["payment_to_income"].values).any()
    assert (augmented["payment_to_income"].values == 0.0).all()

    assert not np.isnan(augmented["loan_to_income"].values).any()
    assert not np.isinf(augmented["loan_to_income"].values).any()
    assert (augmented["loan_to_income"].values == 0.0).all()

    assert not np.isnan(augmented["delinquency_velocity"].values).any()
    assert not np.isinf(augmented["delinquency_velocity"].values).any()
    assert (augmented["delinquency_velocity"].values == 0.0).all()


@settings(max_examples=25, deadline=None)
@given(
    installment=st.floats(min_value=0.0, max_value=2000.0),
    annual_income=st.floats(min_value=0.0, max_value=500000.0),
    loan_amount=st.floats(min_value=0.0, max_value=50000.0),
    delinq_2yrs=st.integers(min_value=0, max_value=20),
    total_acc=st.integers(min_value=0, max_value=100),
)
def test_hypothesis_ratio_invariants(installment, annual_income, loan_amount, delinq_2yrs, total_acc):
    """Property test: Ratios remain non-negative, finite, and bounded for non-negative inputs."""
    pti = calculate_payment_to_income(installment, annual_income)
    lti = calculate_loan_to_income(loan_amount, annual_income)
    vel = calculate_delinquency_velocity(delinq_2yrs, total_acc)

    assert pti >= 0.0 and np.isfinite(pti)
    assert lti >= 0.0 and np.isfinite(lti)
    assert vel >= 0.0 and np.isfinite(vel)


def test_pricing_feature_toggle(synthetic_lending_club_data):
    """Verify include_pricing_features toggle purges interest_rate, grade, sub_grade."""
    df = synthetic_lending_club_data.copy()

    # Disabling pricing features
    X_no_pricing = get_feature_matrix(df, feature_set="enhanced", include_pricing_features=False)
    for pf in PRICING_FEATURES:
        assert pf not in X_no_pricing.columns, f"Pricing feature {pf} found when disabled"

    # Enabling pricing features
    X_with_pricing = get_feature_matrix(df, feature_set="enhanced", include_pricing_features=True)
    for pf in PRICING_FEATURES:
        assert pf in X_with_pricing.columns, f"Pricing feature {pf} missing when enabled"


# =====================================================================
# Preprocessor Tests
# =====================================================================

def test_preprocessor_imputation_and_winsorization():
    """Verify median/mode imputation and exact 99.5th percentile winsorization."""
    # Create dataset with known 99.5th percentile outlier and missing values
    np.random.seed(42)
    n = 1000
    incomes = np.linspace(10000, 100000, n)
    incomes[-1] = 10_000_000.0  # Massive outlier

    df = pd.DataFrame(
        {
            "annual_income": incomes,
            "debt_to_income": np.random.uniform(5, 40, n),
            "home_ownership": ["RENT"] * 500 + ["MORTGAGE"] * 499 + [np.nan],
        }
    )
    df.loc[10, "annual_income"] = np.nan
    df.loc[20, "debt_to_income"] = np.nan

    preprocessor = CreditDataPreprocessor(winsorize_quantile=0.995, scale_numerical=False)
    preprocessor.fit(df)

    # Imputation values
    assert preprocessor.numerical_medians_["annual_income"] > 0
    assert preprocessor.categorical_modes_["home_ownership"] == "RENT"

    # Winsorization cap matches training 99.5th percentile
    expected_cap = df["annual_income"].fillna(preprocessor.numerical_medians_["annual_income"]).quantile(0.995)
    assert np.isclose(preprocessor.winsorize_caps_["annual_income"], expected_cap)

    # Transform
    transformed = preprocessor.transform(df)

    # Assert no NaNs remain
    assert not transformed.isna().any().any()

    # Values at or below 99.5th percentile are unmodified
    normal_idx = 100
    assert np.isclose(transformed.loc[normal_idx, "annual_income"], df.loc[normal_idx, "annual_income"])

    # Outlier is capped at the exact winsorization threshold
    assert transformed.loc[n - 1, "annual_income"] <= preprocessor.winsorize_caps_["annual_income"] + 1e-5


def test_preprocessor_scaling_and_encoding(synthetic_lending_club_data):
    """Verify standard scaling (mean 0, var 1) and one-hot encoding."""
    df = synthetic_lending_club_data.copy()
    X = get_feature_matrix(df, feature_set="baseline", include_pricing_features=False)

    preprocessor = CreditDataPreprocessor(scale_numerical=True)
    preprocessor.fit(X)
    X_trans = preprocessor.transform(X)

    # Numerical features should be standardized
    for col in preprocessor.numerical_cols_:
        mean_val = X_trans[col].mean()
        std_val = X_trans[col].std()
        assert abs(mean_val) < 1e-2, f"Feature {col} mean {mean_val} != 0"
        assert abs(std_val - 1.0) < 1e-2, f"Feature {col} std {std_val} != 1"

    # Categorical features are one-hot encoded
    assert any("home_ownership_" in col for col in X_trans.columns)
    assert any("term_" in col for col in X_trans.columns)


def test_preprocessor_unseen_categories_in_test():
    """Verify preprocessor handles unseen categories in test set gracefully without error."""
    train_df = pd.DataFrame(
        {
            "home_ownership": ["RENT", "MORTGAGE", "RENT"],
            "annual_income": [50000.0, 75000.0, 60000.0],
        }
    )
    test_df = pd.DataFrame(
        {
            "home_ownership": ["UNKNOWN_CATEGORY", "RENT"],
            "annual_income": [55000.0, 65000.0],
        }
    )
    preprocessor = CreditDataPreprocessor()
    preprocessor.fit(train_df)
    test_trans = preprocessor.transform(test_df)

    assert not test_trans.isna().any().any()
    assert len(test_trans) == 2


# =====================================================================
# Partitioning Tests
# =====================================================================

def test_stratified_5fold_splits_distribution(synthetic_lending_club_data):
    """Verify Stratified 5-Fold splits preserve default rate across all folds."""
    df = synthetic_lending_club_data.copy()
    splits = create_stratified_kfold_splits(df, target_column="loan_status", n_splits=5)

    pop_rate = df["loan_status"].mean()
    assert len(splits) == 5

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        val_rate = df.iloc[val_idx]["loan_status"].mean()
        assert abs(val_rate - pop_rate) <= 0.005
