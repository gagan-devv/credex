"""
Shared Pytest Fixtures and Hypothesis Strategies for Credex.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import strategies as st


@pytest.fixture
def synthetic_lending_club_data() -> pd.DataFrame:
    """Generate synthetic LendingClub-like dataframe for testing."""
    np.random.seed(42)
    n_samples = 500

    annual_income = np.random.lognormal(mean=11.0, sigma=0.5, size=n_samples)
    loan_amount = np.random.uniform(1000, 40000, size=n_samples)
    installment = loan_amount / 36.0 + np.random.uniform(10, 50, size=n_samples)
    dti = np.random.uniform(5.0, 45.0, size=n_samples)
    fico_score = np.random.randint(600, 850, size=n_samples)
    delinq_2yrs = np.random.poisson(lam=0.3, size=n_samples)
    revol_util = np.random.uniform(0.0, 100.0, size=n_samples)
    revol_bal = np.random.uniform(500, 30000, size=n_samples)
    total_rev_hi_lim = revol_bal + np.random.uniform(1000, 50000, size=n_samples)
    total_acc = np.random.randint(5, 50, size=n_samples)
    open_acc = np.minimum(total_acc, np.random.randint(2, 25, size=n_samples))
    inq_last_6mths = np.random.poisson(lam=0.5, size=n_samples)
    pub_rec = np.random.poisson(lam=0.1, size=n_samples)
    emp_length = np.random.uniform(0.0, 10.0, size=n_samples)

    home_ownership = np.random.choice(["RENT", "OWN", "MORTGAGE"], size=n_samples)
    term = np.random.choice(["36 months", "60 months"], size=n_samples)
    grade = np.random.choice(["A", "B", "C", "D", "E"], size=n_samples)
    sub_grade = [f"{g}{np.random.randint(1, 6)}" for g in grade]
    interest_rate = np.random.uniform(5.0, 25.0, size=n_samples)

    # Probabilistic default generation based on dti and fico
    logit = (
        -3.0
        + 0.05 * dti
        - 0.01 * (fico_score - 700)
        + 0.3 * delinq_2yrs
        + 0.01 * revol_util
    )
    prob_default = 1.0 / (1.0 + np.exp(-logit))
    loan_status = np.random.binomial(n=1, p=prob_default)

    # Issue dates for OOT testing
    dates = pd.date_range("2012-01-01", "2018-12-31", periods=n_samples)

    df = pd.DataFrame(
        {
            "loan_status": loan_status,
            "issue_d": dates,
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


# Hypothesis Custom Strategies
credit_applicant_strategy = st.fixed_dictionaries(
    {
        "annual_income": st.floats(min_value=10000.0, max_value=500000.0),
        "loan_amount": st.floats(min_value=1000.0, max_value=40000.0),
        "installment": st.floats(min_value=30.0, max_value=1500.0),
        "debt_to_income": st.floats(min_value=0.0, max_value=60.0),
        "fico_score": st.integers(min_value=500, max_value=850),
        "delinq_2yrs": st.integers(min_value=0, max_value=10),
        "revol_utilization": st.floats(min_value=0.0, max_value=100.0),
        "revol_bal": st.floats(min_value=0.0, max_value=100000.0),
        "total_rev_hi_lim": st.floats(min_value=1000.0, max_value=200000.0),
        "total_acc": st.integers(min_value=1, max_value=100),
        "open_acc": st.integers(min_value=1, max_value=50),
        "inq_last_6mths": st.integers(min_value=0, max_value=10),
        "pub_rec": st.integers(min_value=0, max_value=10),
        "emp_length": st.floats(min_value=0.0, max_value=20.0),
        "interest_rate": st.floats(min_value=5.0, max_value=30.0),
    }
)
