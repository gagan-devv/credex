"""
Feature Engineering Engine for Credit Risk Assessment.
Calculates novel domain solvency ratios and isolates econometric endogeneity via pricing feature toggles.
"""

from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import FeatureConfig, load_config


# Canonical pricing features introducing circular endogeneity (PRD Section 3.3)
PRICING_FEATURES = ["interest_rate", "grade", "sub_grade"]

# Canonical engineered solvency ratios (CREDEX-06)
ENGINEERED_RATIOS = [
    "payment_to_income",
    "loan_to_income",
    "liquidity_buffer",
    "delinquency_velocity",
]

# Baseline numerical and categorical columns reproducing Li & Wu (2024) Table 2
BASELINE_NUMERICAL = [
    "debt_to_income",
    "annual_income",
    "loan_amount",
    "installment",
    "interest_rate",
    "fico_score",
    "delinq_2yrs",
    "revol_utilization",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "total_acc",
    "emp_length",
]

BASELINE_CATEGORICAL = [
    "home_ownership",
    "term",
    "grade",
    "sub_grade",
]


def calculate_payment_to_income(
    installment: Union[pd.Series, np.ndarray, float],
    annual_income: Union[pd.Series, np.ndarray, float],
    eps: float = 1e-6,
) -> Union[pd.Series, np.ndarray, float]:
    """
    Calculate Payment-to-Income (PTI) ratio: installment / (annual_income / 12).
    
    Guards against division-by-zero or non-positive incomes.
    """
    installment_arr = np.asarray(installment, dtype=float)
    income_arr = np.asarray(annual_income, dtype=float)

    monthly_income = income_arr / 12.0
    safe_monthly_income = np.where(monthly_income > eps, monthly_income, np.nan)
    pti = np.where(
        (income_arr > 0) & (~np.isnan(income_arr)),
        installment_arr / safe_monthly_income,
        0.0,
    )
    # Replace any leftover NaNs or infs with 0.0
    pti = np.nan_to_num(pti, nan=0.0, posinf=0.0, neginf=0.0)
    
    if isinstance(installment, pd.Series):
        return pd.Series(pti, index=installment.index, name="payment_to_income")
    return pti


def calculate_loan_to_income(
    loan_amount: Union[pd.Series, np.ndarray, float],
    annual_income: Union[pd.Series, np.ndarray, float],
    eps: float = 1e-6,
) -> Union[pd.Series, np.ndarray, float]:
    """
    Calculate Loan-to-Income (LTI) ratio: loan_amount / annual_income.
    
    Guards against division-by-zero or non-positive incomes.
    """
    loan_arr = np.asarray(loan_amount, dtype=float)
    income_arr = np.asarray(annual_income, dtype=float)

    safe_income = np.where(income_arr > eps, income_arr, np.nan)
    lti = np.where(
        (income_arr > 0) & (~np.isnan(income_arr)),
        loan_arr / safe_income,
        0.0,
    )
    lti = np.nan_to_num(lti, nan=0.0, posinf=0.0, neginf=0.0)

    if isinstance(loan_amount, pd.Series):
        return pd.Series(lti, index=loan_amount.index, name="loan_to_income")
    return lti


def calculate_liquidity_buffer(
    total_rev_hi_lim: Union[pd.Series, np.ndarray, float],
    revol_bal: Union[pd.Series, np.ndarray, float],
) -> Union[pd.Series, np.ndarray, float]:
    """
    Calculate Available Liquidity Buffer: total_rev_hi_lim - revol_bal.
    """
    lim_arr = np.asarray(total_rev_hi_lim, dtype=float)
    bal_arr = np.asarray(revol_bal, dtype=float)
    buffer = lim_arr - bal_arr
    buffer = np.nan_to_num(buffer, nan=0.0)

    if isinstance(total_rev_hi_lim, pd.Series):
        return pd.Series(buffer, index=total_rev_hi_lim.index, name="liquidity_buffer")
    return buffer


def calculate_delinquency_velocity(
    delinq_2yrs: Union[pd.Series, np.ndarray, float],
    total_acc: Union[pd.Series, np.ndarray, float],
    eps: float = 1e-6,
) -> Union[pd.Series, np.ndarray, float]:
    """
    Calculate Delinquency Velocity: delinq_2yrs / total_acc.
    
    Guards against division-by-zero when total_acc is zero.
    """
    delinq_arr = np.asarray(delinq_2yrs, dtype=float)
    acc_arr = np.asarray(total_acc, dtype=float)

    safe_acc = np.where(acc_arr > 0, acc_arr, np.nan)
    velocity = np.where(
        (acc_arr > 0) & (~np.isnan(acc_arr)),
        delinq_arr / safe_acc,
        0.0,
    )
    velocity = np.nan_to_num(velocity, nan=0.0, posinf=0.0, neginf=0.0)

    if isinstance(delinq_2yrs, pd.Series):
        return pd.Series(velocity, index=delinq_2yrs.index, name="delinquency_velocity")
    return velocity


def engineer_solvency_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Augment dataframe with 4 core economic solvency ratios:
      1. Payment-to-Income (PTI)
      2. Loan-to-Income (LTI)
      3. Available Liquidity Buffer
      4. Delinquency Velocity
    """
    out_df = df.copy()

    # 1. Payment-to-Income (PTI)
    if "installment" in out_df.columns and "annual_income" in out_df.columns:
        out_df["payment_to_income"] = calculate_payment_to_income(
            out_df["installment"], out_df["annual_income"]
        )

    # 2. Loan-to-Income (LTI)
    if "loan_amount" in out_df.columns and "annual_income" in out_df.columns:
        out_df["loan_to_income"] = calculate_loan_to_income(
            out_df["loan_amount"], out_df["annual_income"]
        )

    # 3. Available Liquidity Buffer
    if "total_rev_hi_lim" in out_df.columns and "revol_bal" in out_df.columns:
        out_df["liquidity_buffer"] = calculate_liquidity_buffer(
            out_df["total_rev_hi_lim"], out_df["revol_bal"]
        )
    elif "revol_bal" in out_df.columns:
        # If total_rev_hi_lim is not present, default buffer to 0.0
        out_df["liquidity_buffer"] = 0.0

    # 4. Delinquency Velocity
    if "delinq_2yrs" in out_df.columns and "total_acc" in out_df.columns:
        out_df["delinquency_velocity"] = calculate_delinquency_velocity(
            out_df["delinq_2yrs"], out_df["total_acc"]
        )

    return out_df


def get_feature_schema(
    feature_set: str = "enhanced",
    include_pricing_features: bool = False,
    config: Optional[FeatureConfig] = None,
) -> Tuple[List[str], List[str]]:
    """
    Retrieve numerical and categorical feature lists according to regime and pricing toggle.
    
    Parameters:
    -----------
    feature_set : str
        'baseline' (Li & Wu 2024 replication) or 'enhanced' (solvency augmented).
    include_pricing_features : bool
        If False, purges interest_rate, grade, sub_grade.
    config : Optional[FeatureConfig]
        Optional parsed feature config schema.
        
    Returns:
    --------
    Tuple[List[str], List[str]]
        (numerical_features, categorical_features)
    """
    if config is not None:
        num_feats = list(config.feature_set_baseline.numerical)
        cat_feats = list(config.feature_set_baseline.categorical)
        ratios = list(config.feature_set_enhanced.engineered_ratios)
        pricing = list(config.feature_set_enhanced.pricing_features)
    else:
        num_feats = list(BASELINE_NUMERICAL)
        cat_feats = list(BASELINE_CATEGORICAL)
        ratios = list(ENGINEERED_RATIOS)
        pricing = list(PRICING_FEATURES)

    if feature_set == "enhanced":
        for r in ratios:
            if r not in num_feats:
                num_feats.append(r)

    if not include_pricing_features:
        num_feats = [c for c in num_feats if c not in pricing]
        cat_feats = [c for c in cat_feats if c not in pricing]

    return num_feats, cat_feats


def get_feature_matrix(
    df: pd.DataFrame,
    feature_set: str = "enhanced",
    include_pricing_features: bool = False,
    config: Optional[FeatureConfig] = None,
) -> pd.DataFrame:
    """
    Extract designated feature subset from dataframe with optional solvency ratio engineering.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input raw or processed dataframe.
    feature_set : str
        'baseline' (exact Li & Wu 2024 replication) or 'enhanced'.
    include_pricing_features : bool
        If False, purges interest_rate, grade, sub_grade.
    config : Optional[FeatureConfig]
        Pydantic FeatureConfig schema.
        
    Returns:
    --------
    pd.DataFrame
        Feature matrix containing strictly requested predictors.
    """
    transformed_df = df.copy()

    # Engineer ratios if enhanced feature set requested and missing
    if feature_set == "enhanced":
        transformed_df = engineer_solvency_ratios(transformed_df)

    num_cols, cat_cols = get_feature_schema(
        feature_set=feature_set,
        include_pricing_features=include_pricing_features,
        config=config,
    )
    target_cols = num_cols + cat_cols

    # Ensure all target columns exist in transformed_df, filling missing with NaN if necessary
    for col in target_cols:
        if col not in transformed_df.columns:
            transformed_df[col] = np.nan

    return transformed_df[target_cols].copy()


class CreditFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-Learn compatible transformer for credit feature engineering and pricing ablation.
    """

    def __init__(
        self,
        feature_set: str = "enhanced",
        include_pricing_features: bool = False,
        config: Optional[FeatureConfig] = None,
    ):
        self.feature_set = feature_set
        self.include_pricing_features = include_pricing_features
        self.config = config
        self.numerical_features_: List[str] = []
        self.categorical_features_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "CreditFeatureEngineer":
        """Identify feature schema."""
        self.numerical_features_, self.categorical_features_ = get_feature_schema(
            feature_set=self.feature_set,
            include_pricing_features=self.include_pricing_features,
            config=self.config,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Engineer solvency ratios and select feature matrix."""
        return get_feature_matrix(
            df=X,
            feature_set=self.feature_set,
            include_pricing_features=self.include_pricing_features,
            config=self.config,
        )

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """Return output column names."""
        return self.numerical_features_ + self.categorical_features_
