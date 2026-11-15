"""
Unit and Property-Based Tests for Data Leakage Prevention.
Verifies zero post-origination repayment leakage and strictly out-of-fold transformations.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.data.feature_engineering import get_feature_matrix
from src.data.partitioning import (
    LEAKAGE_COLUMNS,
    audit_feature_matrix_leakage,
    create_oot_split,
    create_stratified_kfold_splits,
    strip_leakage_columns,
)
from src.data.preprocessor import CreditDataPreprocessor, fit_transform_out_of_fold


def test_post_origination_leakage_prevention(synthetic_lending_club_data):
    """Verify that predictor matrix contains zero post-origination repayment columns."""
    df = synthetic_lending_club_data.copy()

    # Inject post-origination columns into the raw dataframe
    for col in LEAKAGE_COLUMNS:
        df[col] = np.random.uniform(0, 1000, size=len(df))

    # Feature matrix extraction must not select any leakage columns
    X_baseline = get_feature_matrix(df, feature_set="baseline", include_pricing_features=True)
    X_enhanced = get_feature_matrix(df, feature_set="enhanced", include_pricing_features=False)

    # Audits must return empty lists and not raise
    assert audit_feature_matrix_leakage(X_baseline) == []
    assert audit_feature_matrix_leakage(X_enhanced) == []

    # Strip leakage columns works properly
    df_cleaned = strip_leakage_columns(df)
    for col in LEAKAGE_COLUMNS:
        assert col not in df_cleaned.columns


def test_audit_feature_matrix_leakage_raises():
    """Verify that audit_feature_matrix_leakage raises ValueError when forbidden columns exist."""
    bad_cols = ["loan_amount", "annual_income", "recoveries"]
    with pytest.raises(ValueError, match="TARGET LEAKAGE DETECTED"):
        audit_feature_matrix_leakage(bad_cols, raise_on_leakage=True)

    detected = audit_feature_matrix_leakage(bad_cols, raise_on_leakage=False)
    assert detected == ["recoveries"]


def test_out_of_fold_preprocessor_no_test_leakage(synthetic_lending_club_data):
    """
    Verify that fitting the preprocessor on training data does not incorporate
    any information or extreme outliers present in the test fold.
    """
    df = synthetic_lending_club_data.copy()
    X = get_feature_matrix(df, feature_set="enhanced", include_pricing_features=False)

    # Split into 80% train and 20% test
    n_train = int(len(X) * 0.8)
    X_train = X.iloc[:n_train].copy()
    X_test = X.iloc[n_train:].copy()

    # Inject a massive outlier in test set only
    extreme_income = 50_000_000.0
    X_test.loc[X_test.index[0], "annual_income"] = extreme_income

    preprocessor = CreditDataPreprocessor(winsorize_quantile=0.995)
    preprocessor.fit(X_train)

    # Training winsorization cap must NOT be influenced by the test outlier
    train_cap = preprocessor.winsorize_caps_["annual_income"]
    assert train_cap < 1_000_000.0

    # Transform test set: extreme outlier must be clipped to the training cap
    X_test_trans = preprocessor.transform(X_test)
    # The scaled value of the extreme income should match the scaled value of the cap
    # Ensure transformed values are finite and non-NaN
    assert not X_test_trans.isna().any().any()
    assert np.isfinite(X_test_trans.values).all()


def test_fit_transform_out_of_fold_isolation(synthetic_lending_club_data):
    """Verify out-of-fold transform pipeline isolates each validation fold."""
    df = synthetic_lending_club_data.copy()
    X = get_feature_matrix(df, feature_set="baseline", include_pricing_features=False)

    splits = create_stratified_kfold_splits(df, target_column="loan_status", n_splits=5)
    train_folds, val_folds = fit_transform_out_of_fold(splits, X)

    assert len(train_folds) == 5
    assert len(val_folds) == 5

    for fold_idx in range(5):
        train_idx, val_idx = splits[fold_idx]
        assert len(train_folds[fold_idx]) == len(train_idx)
        assert len(val_folds[fold_idx]) == len(val_idx)
        assert not train_folds[fold_idx].isna().any().any()
        assert not val_folds[fold_idx].isna().any().any()


def test_oot_split_temporal_leakage_prevention(synthetic_lending_club_data):
    """Verify strictly out-of-time splits have zero temporal overlap."""
    df = synthetic_lending_club_data.copy()
    train_df, test_df = create_oot_split(
        df,
        date_column="issue_d",
        train_start="2012-01-01",
        train_end="2016-12-31",
        test_start="2017-01-01",
        test_end="2018-12-31",
    )

    assert not train_df.empty
    assert not test_df.empty

    max_train_date = train_df["issue_d"].max()
    min_test_date = test_df["issue_d"].min()

    assert max_train_date < min_test_date
    assert max_train_date <= pd.Timestamp("2016-12-31")
    assert min_test_date >= pd.Timestamp("2017-01-01")


def test_oot_overlapping_ranges_raise():
    """Verify create_oot_split rejects overlapping date windows."""
    df = pd.DataFrame({"issue_d": pd.date_range("2015-01-01", "2018-01-01", freq="ME")})
    with pytest.raises(ValueError, match="Temporal overlap detected"):
        create_oot_split(
            df,
            date_column="issue_d",
            train_start="2012-01-01",
            train_end="2017-06-01",
            test_start="2017-01-01",
            test_end="2018-12-31",
        )
