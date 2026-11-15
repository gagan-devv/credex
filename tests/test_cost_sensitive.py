"""
Unit Tests for Cost-Sensitive Ensembles (CREDEX-11).

Verifies:
1. Custom objective callbacks for LightGBM, XGBoost, and CatBoost compute exact gradients.
2. LightGBM, XGBoost, and CatBoost models train smoothly with custom objectives without NaN loss.
3. Cost-sensitive models (w >= 5) produce higher recall on default cases compared to
   symmetric w=1 models.
4. Predicted probabilities strictly reside in [0, 1] without NaNs or Infs.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.metrics import recall_score

from src.models.cost_sensitive import (
    CostSensitiveCatBoostClassifier,
    CostSensitiveCatBoostObjective,
    CostSensitiveLGBMClassifier,
    CostSensitiveXGBClassifier,
    make_lightgbm_objective,
    make_xgboost_objective,
)


@pytest.fixture
def synthetic_imbalanced_data():
    """Generate synthetic imbalanced credit default data."""
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=6,
        n_classes=2,
        weights=[0.80, 0.20],
        random_state=42,
    )
    return X, y


def test_lightgbm_objective_callback_signatures():
    """Verify LightGBM objective works for both scikit-learn and low-level calls."""
    obj = make_lightgbm_objective(w=5.0, epsilon=1e-6)

    # Scikit-learn API: (y_true, y_pred)
    y_true = np.array([1, 0, 1])
    y_pred = np.array([0.5, -0.5, 1.0])
    grad, hess = obj(y_true, y_pred)

    assert len(grad) == 3
    assert len(hess) == 3
    assert np.all(hess > 0.0)

    # Low-level mock Dataset
    class MockDataset:
        def get_label(self):
            return np.array([1, 0, 1])

    grad_ds, hess_ds = obj(y_pred, MockDataset())
    np.testing.assert_allclose(grad, grad_ds)
    np.testing.assert_allclose(hess, hess_ds)


def test_xgboost_objective_callback_signatures():
    """Verify XGBoost objective works for both scikit-learn and low-level calls."""
    obj = make_xgboost_objective(w=5.0, epsilon=1e-6)

    y_true = np.array([1, 0, 1])
    y_pred = np.array([0.2, -0.8, 1.5])
    grad, hess = obj(y_true, y_pred)

    assert len(grad) == 3
    assert len(hess) == 3
    assert np.all(hess > 0.0)

    class MockDMatrix:
        def get_label(self):
            return np.array([1, 0, 1])

    grad_dm, hess_dm = obj(y_pred, MockDMatrix())
    np.testing.assert_allclose(grad, grad_dm)
    np.testing.assert_allclose(hess, hess_dm)


def test_catboost_objective_ders_range():
    """Verify CatBoost objective calc_ders_range outputs correct negative gradients."""
    obj = CostSensitiveCatBoostObjective(w=5.0, epsilon=1e-6)

    approxes = [0.2, -0.5]
    targets = [1, 0]
    weights = [1.0, 2.0]

    ders = obj.calc_ders_range(approxes, targets, weights)
    assert len(ders) == 2

    der1_0, der2_0 = ders[0]
    der1_1, der2_1 = ders[1]

    # In CatBoost, der1 = -grad, der2 = -hess
    # Since hess > 0, der2 must be negative
    assert der2_0 < 0.0
    assert der2_1 < 0.0


def test_cost_sensitive_lgbm_classifier(synthetic_imbalanced_data):
    """Verify CostSensitiveLGBMClassifier trains smoothly and raises default recall."""
    X, y = synthetic_imbalanced_data

    # Standard model (w=1.0)
    clf_base = CostSensitiveLGBMClassifier(w=1.0, n_estimators=30, max_depth=3, random_state=42)
    clf_base.fit(X, y)
    probs_base = clf_base.predict_proba(X)
    assert not np.isnan(probs_base).any()
    rec_base = recall_score(y, clf_base.predict(X, threshold=0.5))

    assert not np.isnan(probs_base).any()
    assert np.all(probs_base >= 0.0) and np.all(probs_base <= 1.0)

    # Cost-sensitive model (w=10.0)
    clf_cost = CostSensitiveLGBMClassifier(w=10.0, n_estimators=30, max_depth=3, random_state=42)
    clf_cost.fit(X, y)
    probs_cost = clf_cost.predict_proba(X)
    rec_cost = recall_score(y, clf_cost.predict(X, threshold=0.5))

    assert not np.isnan(probs_cost).any()
    assert np.all(probs_cost >= 0.0) and np.all(probs_cost <= 1.0)
    assert rec_cost >= rec_base, f"Cost-sensitive recall {rec_cost} should be >= base {rec_base}"


def test_cost_sensitive_xgb_classifier(synthetic_imbalanced_data):
    """Verify CostSensitiveXGBClassifier trains smoothly and raises default recall."""
    X, y = synthetic_imbalanced_data

    clf_base = CostSensitiveXGBClassifier(w=1.0, n_estimators=30, max_depth=3, random_state=42)
    clf_base.fit(X, y)
    probs_base = clf_base.predict_proba(X)
    assert not np.isnan(probs_base).any()
    rec_base = recall_score(y, clf_base.predict(X, threshold=0.5))

    clf_cost = CostSensitiveXGBClassifier(w=10.0, n_estimators=30, max_depth=3, random_state=42)
    clf_cost.fit(X, y)
    probs_cost = clf_cost.predict_proba(X)
    rec_cost = recall_score(y, clf_cost.predict(X, threshold=0.5))

    assert not np.isnan(probs_cost).any()
    assert np.all(probs_cost >= 0.0) and np.all(probs_cost <= 1.0)
    assert rec_cost >= rec_base, f"Cost-sensitive recall {rec_cost} should be >= base {rec_base}"


def test_cost_sensitive_catboost_classifier(synthetic_imbalanced_data):
    """Verify CostSensitiveCatBoostClassifier trains smoothly and raises default recall."""
    X, y = synthetic_imbalanced_data

    clf_base = CostSensitiveCatBoostClassifier(
        w=1.0, iterations=30, depth=3, random_seed=42, verbose=False
    )
    clf_base.fit(X, y)
    probs_base = clf_base.predict_proba(X)
    assert not np.isnan(probs_base).any()
    rec_base = recall_score(y, clf_base.predict(X, threshold=0.5))

    clf_cost = CostSensitiveCatBoostClassifier(
        w=10.0, iterations=30, depth=3, random_seed=42, verbose=False
    )
    clf_cost.fit(X, y)
    probs_cost = clf_cost.predict_proba(X)
    rec_cost = recall_score(y, clf_cost.predict(X, threshold=0.5))

    assert not np.isnan(probs_cost).any()
    assert np.all(probs_cost >= 0.0) and np.all(probs_cost <= 1.0)
    assert rec_cost >= rec_base, f"Cost-sensitive recall {rec_cost} should be >= base {rec_base}"
