"""
Partitioning Engine & Target Leakage Prevention.
Generates Stratified 5-Fold Cross-Validation splits and Out-of-Time (OOT) holdouts.
Audits and purges post-origination repayment leakage columns.
"""

from typing import Iterable, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


# Programmatic registry of post-origination repayment fields (PRD Section 3.4 & CREDEX-07)
LEAKAGE_COLUMNS: Set[str] = {
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_amnt",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "last_pymnt_d",
    "next_pymnt_d",
    "last_credit_pull_d",
    "out_prncp",
    "out_prncp_inv",
    "total_pymnt",
    "total_pymnt_inv",
    "settlement_status",
    "settlement_date",
    "settlement_amount",
    "settlement_percentage",
    "settlement_term",
    "debt_settlement_flag",
    "hardship_flag",
}


def audit_feature_matrix_leakage(
    features_or_df: Union[Iterable[str], pd.DataFrame],
    raise_on_leakage: bool = True,
) -> List[str]:
    """
    Programmatically verify that zero post-origination columns appear in the predictor matrix.
    
    Parameters:
    -----------
    features_or_df : Union[Iterable[str], pd.DataFrame]
        List of feature names or a DataFrame to audit.
    raise_on_leakage : bool
        If True, raises ValueError when leakage columns are detected.
        
    Returns:
    --------
    List[str]
        List of identified leaking column names.
    """
    if isinstance(features_or_df, pd.DataFrame):
        column_set = set(features_or_df.columns)
    else:
        column_set = set(features_or_df)

    leaking_cols = sorted(list(column_set.intersection(LEAKAGE_COLUMNS)))

    if leaking_cols and raise_on_leakage:
        raise ValueError(
            f"TARGET LEAKAGE DETECTED: The following post-origination columns were found in the "
            f"predictor matrix: {leaking_cols}. These must be purged before model training."
        )

    return leaking_cols


def strip_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Purge all post-origination repayment fields from dataframe."""
    cols_to_drop = [c for c in df.columns if c in LEAKAGE_COLUMNS]
    if cols_to_drop:
        return df.drop(columns=cols_to_drop)
    return df.copy()


def create_stratified_kfold_splits(
    df: pd.DataFrame,
    target_column: str = "loan_status",
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate Stratified K-Fold Cross-Validation splits.
    
    Preserves exact default rate across all folds reproducing Li & Wu (2024).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing target_column.
    target_column : str
        Name of target column.
    n_splits : int
        Number of folds (default 5).
    shuffle : bool
        Whether to shuffle data before splitting.
    random_state : int
        Random seed for reproducibility.
        
    Returns:
    --------
    List[Tuple[np.ndarray, np.ndarray]]
        List of (train_indices, val_indices) tuples.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe.")

    y = df[target_column].values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    splits = list(skf.split(df, y))

    # Validate that default rates in every validation fold align with population within +/- 0.5%
    pop_default_rate = np.mean(y)
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        val_default_rate = np.mean(y[val_idx])
        diff = abs(val_default_rate - pop_default_rate)
        assert diff <= 0.005, (
            f"Fold {fold_idx} default rate {val_default_rate:.4f} deviates from population "
            f"{pop_default_rate:.4f} by {diff:.4f} (> 0.5%)"
        )

    return splits


def create_oot_split(
    df: pd.DataFrame,
    date_column: str = "issue_d",
    train_start: str = "2012-01-01",
    train_end: str = "2016-12-31",
    test_start: str = "2017-01-01",
    test_end: str = "2018-12-31",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate strictly Out-of-Time (OOT) holdout partitions.
    
    Development window: [train_start, train_end] (e.g. 2012-2016)
    Evaluation window:  [test_start, test_end]   (e.g. 2017-2018)
    
    Guarantees zero temporal overlap between development and evaluation cohorts.
    """
    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' not found in dataframe.")

    date_series = pd.to_datetime(df[date_column])

    ts_train_start = pd.Timestamp(train_start)
    ts_train_end = pd.Timestamp(train_end)
    ts_test_start = pd.Timestamp(test_start)
    ts_test_end = pd.Timestamp(test_end)

    if ts_train_end >= ts_test_start:
        raise ValueError(
            f"Temporal overlap detected: train_end ({ts_train_end}) >= test_start ({ts_test_start})"
        )

    train_mask = (date_series >= ts_train_start) & (date_series <= ts_train_end)
    test_mask = (date_series >= ts_test_start) & (date_series <= ts_test_end)

    train_df = df[train_mask].copy().reset_index(drop=True)
    test_df = df[test_mask].copy().reset_index(drop=True)

    # Invariant: zero temporal overlap
    if not train_df.empty and not test_df.empty:
        max_train_date = pd.to_datetime(train_df[date_column]).max()
        min_test_date = pd.to_datetime(test_df[date_column]).min()
        assert max_train_date < min_test_date, (
            f"Temporal leakage detected: max train date {max_train_date} >= min test date {min_test_date}"
        )

    return train_df, test_df
