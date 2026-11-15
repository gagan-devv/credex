"""
Data Ingestion Engine for LendingClub and South German Credit Datasets.
Supports completed contract filtering, dual-scale execution, and deterministic stratified sampling.
"""

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Completed contract status definitions (Li & Wu 2024 / PRD Section 3.1)
COMPLETED_LOAN_STATUSES = ["Fully Paid", "Charged Off", "Default"]

TARGET_MAPPING: Dict[str, int] = {
    "Fully Paid": 0,
    "Charged Off": 1,
    "Default": 1,
    "Does not meet the credit policy. Status:Fully Paid": 0,
    "Does not meet the credit policy. Status:Charged Off": 1,
}

# Raw LendingClub column name aliases to standardize to Credex feature schema
LENDING_CLUB_COLUMN_MAP: Dict[str, str] = {
    "loan_amnt": "loan_amount",
    "dti": "debt_to_income",
    "revol_util": "revol_utilization",
    "int_rate": "interest_rate",
}

# Global in-memory cache
_LOADER_CACHE: Dict[str, pd.DataFrame] = {}


def clear_loader_cache() -> None:
    """Clear in-memory cache of loaded datasets."""
    _LOADER_CACHE.clear()


def clean_percentage(val: Union[str, float, int]) -> float:
    """Parse percentage strings (e.g., '12.5%') into float (12.5)."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        val_str = val.strip().rstrip("%")
        try:
            return float(val_str)
        except ValueError:
            return np.nan
    return np.nan


def clean_emp_length(val: Union[str, float, int]) -> float:
    """Parse emp_length strings ('< 1 year', '10+ years', '3 years') to float."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    val_str = str(val).strip()
    if "<" in val_str:
        return 0.5
    if "10" in val_str:
        return 10.0
    digits = "".join([c for c in val_str if c.isdigit()])
    if digits:
        return float(digits)
    return np.nan


def filter_completed_loans(
    df: pd.DataFrame,
    status_column: str = "loan_status",
    target_column: str = "loan_status",
) -> pd.DataFrame:
    """
    Filter loan dataset to completed loan contracts with definitive terminal performance.
    
    Rejects active, current, late, or in-grace-period contracts.
    Maps:
      'Fully Paid' -> 0
      'Charged Off' / 'Default' -> 1
    """
    if status_column not in df.columns:
        raise ValueError(f"Status column '{status_column}' not found in dataframe.")

    filtered_df = df.copy()

    # If already numeric binary {0, 1}, validate and keep
    if pd.api.types.is_numeric_dtype(filtered_df[status_column]):
        unique_vals = set(filtered_df[status_column].dropna().unique())
        if unique_vals.issubset({0, 1}):
            filtered_df[target_column] = filtered_df[status_column].astype(int)
            return filtered_df
        raise ValueError(f"Numeric status column has non-binary values: {unique_vals}")

    # String status filtering
    status_series = filtered_df[status_column].astype(str).str.strip()
    mask = status_series.isin(TARGET_MAPPING.keys())
    filtered_df = filtered_df[mask].copy()

    # Map target
    filtered_df[target_column] = status_series[mask].map(TARGET_MAPPING).astype(int)
    return filtered_df


def standardize_lending_club_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize raw LendingClub column names and clean string encodings."""
    standardized = df.rename(columns=LENDING_CLUB_COLUMN_MAP).copy()

    # Calculate fico_score from fico_range_low and fico_range_high if needed
    if "fico_score" not in standardized.columns:
        if "fico_range_low" in standardized.columns and "fico_range_high" in standardized.columns:
            standardized["fico_score"] = (
                standardized["fico_range_low"] + standardized["fico_range_high"]
            ) / 2.0

    # Clean percentages
    for pct_col in ["interest_rate", "revol_utilization"]:
        if pct_col in standardized.columns and standardized[pct_col].dtype == object:
            standardized[pct_col] = standardized[pct_col].apply(clean_percentage)

    # Clean emp_length
    if "emp_length" in standardized.columns and standardized["emp_length"].dtype == object:
        standardized["emp_length"] = standardized["emp_length"].apply(clean_emp_length)

    # Standardize term
    if "term" in standardized.columns and standardized["term"].dtype == object:
        standardized["term"] = standardized["term"].astype(str).str.strip()

    # Parse issue_d
    if "issue_d" in standardized.columns:
        if not pd.api.types.is_datetime64_any_dtype(standardized["issue_d"]):
            standardized["issue_d"] = pd.to_datetime(standardized["issue_d"], errors="coerce")

    return standardized


def generate_synthetic_lending_club(
    n_samples: int = 200000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate high-fidelity synthetic LendingClub dataset satisfying PRD schemas."""
    rng = np.random.default_rng(random_state)

    annual_income = rng.lognormal(mean=11.0, sigma=0.5, size=n_samples)
    loan_amount = rng.uniform(1000.0, 40000.0, size=n_samples)
    installment = loan_amount / 36.0 + rng.uniform(10.0, 50.0, size=n_samples)
    dti = rng.uniform(5.0, 45.0, size=n_samples)
    fico_score = rng.integers(600, 850, size=n_samples)
    delinq_2yrs = rng.poisson(lam=0.3, size=n_samples)
    revol_util = rng.uniform(0.0, 100.0, size=n_samples)
    revol_bal = rng.uniform(500.0, 30000.0, size=n_samples)
    total_rev_hi_lim = revol_bal + rng.uniform(1000.0, 50000.0, size=n_samples)
    total_acc = rng.integers(5, 50, size=n_samples)
    open_acc = np.minimum(total_acc, rng.integers(2, 25, size=n_samples))
    inq_last_6mths = rng.poisson(lam=0.5, size=n_samples)
    pub_rec = rng.poisson(lam=0.1, size=n_samples)
    emp_length = rng.uniform(0.0, 10.0, size=n_samples)

    home_ownership = rng.choice(["RENT", "OWN", "MORTGAGE"], size=n_samples)
    term = rng.choice(["36 months", "60 months"], size=n_samples)
    grade = rng.choice(["A", "B", "C", "D", "E"], size=n_samples)
    sub_grade = [f"{g}{rng.integers(1, 6)}" for g in grade]
    interest_rate = rng.uniform(5.0, 25.0, size=n_samples)

    # Empirical credit default logistic generator (~20% base default rate)
    logit = (
        -3.0
        + 0.05 * dti
        - 0.01 * (fico_score - 700)
        + 0.3 * delinq_2yrs
        + 0.01 * revol_util
    )
    prob_default = 1.0 / (1.0 + np.exp(-logit))
    loan_status = rng.binomial(n=1, p=prob_default)

    # Issue dates spanning 2012-01-01 to 2018-12-31 for OOT testing
    start_date = pd.Timestamp("2012-01-01")
    end_date = pd.Timestamp("2018-12-31")
    delta_days = (end_date - start_date).days
    random_days = rng.integers(0, delta_days + 1, size=n_samples)
    issue_dates = [start_date + pd.Timedelta(days=int(d)) for d in random_days]

    df = pd.DataFrame(
        {
            "loan_status": loan_status,
            "issue_d": issue_dates,
            "annual_income": annual_income,
            "loan_amount": loan_amount,
            "installment": installment,
            "debt_to_income": dti,
            "fico_score": fico_score,
            "delinq_2yrs": delinq_2yrs,
            "revol_utilization": revol_util,
            "revol_bal": revol_bal,
            "total_rev_hi_lim": total_rev_hi_lim,
            "total_acc": total_acc,
            "open_acc": open_acc,
            "inq_last_6mths": inq_last_6mths,
            "pub_rec": pub_rec,
            "emp_length": emp_length,
            "home_ownership": home_ownership,
            "term": term,
            "grade": grade,
            "sub_grade": sub_grade,
            "interest_rate": interest_rate,
        }
    )
    return df


def load_lending_club(
    path: Optional[str] = None,
    sample_size: int = 200000,
    use_full_dataset: bool = False,
    random_state: int = 42,
    fallback_to_synthetic: bool = True,
    n_synthetic: int = 50000,
) -> pd.DataFrame:
    """
    Ingest LendingClub loan data with completed contract filtering and dual-mode sampling.
    
    Parameters:
    -----------
    path : Optional[str]
        File path to raw LendingClub CSV/parquet. If None, checks configs or synthetic.
    sample_size : int
        Cohort size for stratified development sample (default 200,000).
    use_full_dataset : bool
        If True, loads complete dataset without subsampling.
    random_state : int
        Seed for deterministic stratified sampling.
    fallback_to_synthetic : bool
        If True and file does not exist, generates synthetic dataset.
    n_synthetic : int
        Number of synthetic samples to generate if falling back.
        
    Returns:
    --------
    pd.DataFrame
        Filtered, standardized LendingClub dataset.
    """
    cache_key = f"lc_{path}_{sample_size}_{use_full_dataset}_{random_state}_{fallback_to_synthetic}_{n_synthetic}"
    if cache_key in _LOADER_CACHE:
        return _LOADER_CACHE[cache_key].copy()

    df: Optional[pd.DataFrame] = None

    if path is not None and Path(path).is_file():
        file_path = Path(path)
        if file_path.suffix.lower() == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path, low_memory=False)
    elif fallback_to_synthetic:
        gen_n = max(sample_size, n_synthetic)
        df = generate_synthetic_lending_club(n_samples=gen_n, random_state=random_state)
    else:
        raise FileNotFoundError(f"LendingClub data file not found at: {path}")

    # Standardize column names and format
    df = standardize_lending_club_columns(df)

    # Filter completed loans and map binary target
    df = filter_completed_loans(df, status_column="loan_status", target_column="loan_status")

    # Dual-mode sampling
    if not use_full_dataset and len(df) > sample_size:
        # Deterministic stratified sample maintaining default rate
        stratify_target = df["loan_status"]
        df_sampled, _ = train_test_split(
            df,
            train_size=sample_size,
            stratify=stratify_target,
            random_state=random_state,
        )
        df = df_sampled.reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    _LOADER_CACHE[cache_key] = df
    return df.copy()


def generate_synthetic_german_credit(
    n_samples: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate synthetic South German Credit dataset conforming to UCI benchmark."""
    rng = np.random.default_rng(random_state)

    duration = rng.integers(4, 72, size=n_samples)
    amount = rng.uniform(250.0, 18000.0, size=n_samples)
    installment_rate = rng.integers(1, 5, size=n_samples)
    present_residence = rng.integers(1, 5, size=n_samples)
    age = rng.integers(19, 75, size=n_samples)
    number_credits = rng.integers(1, 5, size=n_samples)
    people_liable = rng.choice([1, 2], p=[0.85, 0.15], size=n_samples)

    status = rng.choice(["A11", "A12", "A13", "A14"], size=n_samples)
    credit_history = rng.choice(["A30", "A31", "A32", "A33", "A34"], size=n_samples)
    purpose = rng.choice(["A40", "A41", "A42", "A43", "A44", "A45", "A46", "A47", "A48", "A49", "A410"], size=n_samples)
    savings = rng.choice(["A61", "A62", "A63", "A64", "A65"], size=n_samples)
    employment_duration = rng.choice(["A71", "A72", "A73", "A74", "A75"], size=n_samples)
    personal_status_sex = rng.choice(["A91", "A92", "A93", "A94", "A95"], size=n_samples)
    other_debtors = rng.choice(["A101", "A102", "A103"], size=n_samples)
    property_type = rng.choice(["A121", "A122", "A123", "A124"], size=n_samples)
    other_installment_plans = rng.choice(["A141", "A142", "A143"], size=n_samples)
    housing = rng.choice(["A151", "A152", "A153"], size=n_samples)
    job = rng.choice(["A171", "A172", "A173", "A174"], size=n_samples)
    telephone = rng.choice(["A191", "A192"], size=n_samples)
    foreign_worker = rng.choice(["A201", "A202"], p=[0.96, 0.04], size=n_samples)

    # 30% default rate standard in South German Credit benchmark
    logit = (
        -1.5
        + 0.02 * duration
        + 0.0001 * amount
        - 0.02 * (age - 35)
        + 0.2 * installment_rate
    )
    prob_default = 1.0 / (1.0 + np.exp(-logit))
    credit_risk = rng.binomial(n=1, p=prob_default)

    df = pd.DataFrame(
        {
            "status": status,
            "duration": duration,
            "credit_history": credit_history,
            "purpose": purpose,
            "amount": amount,
            "savings": savings,
            "employment_duration": employment_duration,
            "installment_rate": installment_rate,
            "personal_status_sex": personal_status_sex,
            "other_debtors": other_debtors,
            "present_residence": present_residence,
            "property": property_type,
            "age": age,
            "other_installment_plans": other_installment_plans,
            "housing": housing,
            "number_credits": number_credits,
            "job": job,
            "people_liable": people_liable,
            "telephone": telephone,
            "foreign_worker": foreign_worker,
            "credit_risk": credit_risk,
            "loan_status": credit_risk,  # Standardized target column
        }
    )
    return df


def load_german_credit(
    path: Optional[str] = None,
    fallback_to_synthetic: bool = True,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Ingest UCI South German Credit dataset with standardized binary credit risk target.
    
    Target:
      1 = Default / Bad Credit Risk
      0 = Non-Default / Good Credit Risk
    """
    cache_key = f"german_{path}_{random_state}_{fallback_to_synthetic}"
    if cache_key in _LOADER_CACHE:
        return _LOADER_CACHE[cache_key].copy()

    df: Optional[pd.DataFrame] = None
    if path is not None and Path(path).is_file():
        file_path = Path(path)
        if file_path.suffix.lower() == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path)
    elif fallback_to_synthetic:
        df = generate_synthetic_german_credit(n_samples=1000, random_state=random_state)
    else:
        raise FileNotFoundError(f"German Credit data file not found at: {path}")

    # Standardize target column: 1 = bad/default, 0 = good
    if "credit_risk" in df.columns and "loan_status" not in df.columns:
        df["loan_status"] = df["credit_risk"]
    elif "kredit" in df.columns and "loan_status" not in df.columns:
        # In original German credit, kredit=1 is good, 0 is bad -> invert to 1=bad
        unique_kredit = set(df["kredit"].unique())
        if unique_kredit == {0, 1}:
            df["loan_status"] = 1 - df["kredit"]
        else:
            df["loan_status"] = df["kredit"]

    _LOADER_CACHE[cache_key] = df
    return df.copy()
