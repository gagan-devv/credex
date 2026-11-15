"""
Unit and Property Tests for TreeSHAP Attributions and Visualizations (CREDEX-16).
"""

import numpy as np
import pandas as pd
import pytest
from src.explainability.shap_explainer import TreeSHAPExplainer, TreeSHAPResult
from src.models.baselines import BaselineModel


@pytest.fixture
def trained_tree_model(synthetic_lending_club_data):
    """Fixture delivering a trained Random Forest model on synthetic data."""
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

    model = BaselineModel(model_name="random_forest", n_estimators=20, max_depth=6, random_state=42)
    model.fit(X, y)
    return model, X, y


def test_shap_explainer_initialization_and_explain(trained_tree_model):
    """Verify TreeSHAP explainer initializes and computes attributions."""
    model, X, y = trained_tree_model
    explainer = TreeSHAPExplainer(model=model, feature_names=list(X.columns))
    
    # Compute local SHAP values on subset
    result = explainer.explain(X.iloc[:50], check_additivity=True)
    assert isinstance(result, TreeSHAPResult)
    assert result.shap_values.shape == (50, len(X.columns))
    assert not np.isnan(result.shap_values).any()
    assert result.is_additive


def test_shap_additivity_property(trained_tree_model):
    """Property test: TreeSHAP values satisfy additive efficiency phi_0 + sum(phi_j) == f(x)."""
    model, X, y = trained_tree_model
    explainer = TreeSHAPExplainer(model=model, feature_names=list(X.columns))
    
    # Test on 20 samples
    X_sub = X.iloc[:20]
    result = explainer.explain(X_sub, check_additivity=True, tolerance=1e-3)
    
    assert result.is_additive
    assert result.max_additivity_error < 1e-3
    assert len(result.base_values) == 20
    assert len(result.model_outputs) == 20


def test_global_importance_ranking(trained_tree_model):
    """Verify global feature importance extraction matches mean absolute SHAP."""
    model, X, y = trained_tree_model
    explainer = TreeSHAPExplainer(model=model, feature_names=list(X.columns))
    
    result = explainer.explain(X.iloc[:100])
    importance_series = result.global_importance
    assert isinstance(importance_series, pd.Series)
    assert len(importance_series) == len(X.columns)
    
    # Importance values should be non-negative and sorted descending
    vals = importance_series.values
    assert (vals >= 0).all()
    assert np.all(np.diff(vals) <= 0)
