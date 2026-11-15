"""
Unit and Property Tests for Constrained Counterfactual Recourse & Adverse Action (CREDEX-17).
"""

import numpy as np
import pandas as pd
import pytest
from src.explainability.adverse_action import (
    DEFAULT_IMMUTABLE_FEATURES,
    DEFAULT_MUTABLE_FEATURES,
    ActionableRecourseEngine,
    ActionableRecourseResult,
    RecourseOption,
)
from src.models.baselines import BaselineModel


@pytest.fixture
def trained_scoring_model(synthetic_lending_club_data):
    """Fixture delivering a trained model and high-risk applicant."""
    df = synthetic_lending_club_data
    feature_cols = [
        "debt_to_income",
        "annual_income",
        "loan_amount",
        "installment",
        "fico_score",
        "delinq_2yrs",
        "revol_utilization",
        "revol_bal",
        "total_acc",
        "emp_length",
    ]
    X = df[feature_cols].copy()
    y = df["loan_status"].values

    model = BaselineModel(model_name="random_forest", n_estimators=30, max_depth=6, random_state=42)
    model.fit(X, y)
    return model, X, y


def test_recourse_engine_immutable_invariance(trained_scoring_model):
    """Property test: Immutable features remain strictly unchanged in all solved recourses."""
    model, X, y = trained_scoring_model
    engine = ActionableRecourseEngine(
        model=model,
        feature_names=list(X.columns),
        tau_star=0.25,
    )

    # Pick a high risk applicant
    probs = model.predict_proba(X)[:, 1]
    high_risk_idx = np.argmax(probs)
    applicant = X.iloc[high_risk_idx]
    current_p = probs[high_risk_idx]
    target_cutoff = max(0.15, current_p - 0.20)

    result = engine.generate_recourse(
        applicant=applicant,
        tau_star=target_cutoff,
        applicant_id="TEST-APP-001",
        compute_shap=False,
    )
    assert isinstance(result, ActionableRecourseResult)

    if result.is_feasible and not result.is_approved:
        # Check that recourse_features preserves all immutable features
        recourse_profile = result.recourse_features
        for immut_col in DEFAULT_IMMUTABLE_FEATURES:
            if immut_col in applicant.index:
                assert np.isclose(
                    float(applicant[immut_col]),
                    float(recourse_profile[immut_col]),
                    atol=1e-6,
                ), f"Immutable feature {immut_col} was altered!"


def test_adverse_action_notice_generation(trained_scoring_model):
    """Verify ECOA Regulation B compliant adverse action disclosure notice formatting."""
    model, X, y = trained_scoring_model
    engine = ActionableRecourseEngine(
        model=model,
        feature_names=list(X.columns),
        tau_star=0.20,
    )

    probs = model.predict_proba(X)[:, 1]
    high_risk_idx = np.argmax(probs)
    applicant = X.iloc[high_risk_idx]

    result = engine.generate_recourse(
        applicant=applicant,
        tau_star=0.20,
        applicant_id="TEST-APP-9901",
        compute_shap=True,
    )

    assert "ADVERSE ACTION DISCLOSURE NOTICE" in result.adverse_action_notice
    assert "REGULATION B / ECOA" in result.adverse_action_notice
    assert "TEST-APP-9901" in result.adverse_action_notice
    assert len(result.top_adverse_factors) >= 1
    assert result.execution_time_ms < 500.0  # Fast solver constraint
