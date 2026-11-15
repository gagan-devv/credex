"""
Comprehensive Unit Tests and Hypothesis Property-Based Tests for Lane C.

Verifies:
  1. All 9 baseline algorithms replicate scikit-learn estimator interface (fit, predict, predict_proba).
  2. Probability calibration engine (Isotonic Regression & Platt Scaling).
  3. Out-of-fold calibration pipeline.
  4. Brier score loss, Expected Calibration Error (ECE), and calibration curve calculations.
  5. Property-based invariants: calibrated probabilities remain strictly within [0, 1],
     isotonic monotonicity, valid binary predictions, and row probability normalization.
"""

from typing import List
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st
from sklearn.metrics import brier_score_loss

from src.models.baselines import (
    BASELINE_MODEL_NAMES,
    DEFAULT_BASELINE_PARAMS,
    PUBLISHED_BENCHMARKS,
    BaselineModel,
    get_all_baseline_models,
    get_baseline_model,
    run_baseline_replication,
)
from src.models.calibration import (
    OutOfFoldCalibrator,
    ProbabilityCalibrator,
    compute_brier_score,
    compute_calibration_curve,
    compute_expected_calibration_error,
    evaluate_calibration,
)

# Numeric predictor columns from synthetic lending club data
FEATURE_COLS = [
    "annual_income",
    "loan_amount",
    "installment",
    "debt_to_income",
    "fico_score",
    "delinq_2yrs",
    "revol_utilization",
    "revol_bal",
    "total_rev_hi_lim",
    "total_acc",
    "open_acc",
    "inq_last_6mths",
    "pub_rec",
    "emp_length",
    "interest_rate",
]


# =====================================================================
# Unit Tests: Baseline Estimators (CREDEX-08)
# =====================================================================


def test_baseline_model_names_count():
    """Verify all 9 canonical algorithms from Li & Wu (2024) are registered."""
    assert len(BASELINE_MODEL_NAMES) == 9
    expected = {
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "adaboost",
        "gbdt",
        "xgboost",
        "lightgbm",
        "catboost",
        "mlp",
    }
    assert set(BASELINE_MODEL_NAMES) == expected


def test_get_baseline_model_factory():
    """Verify factory returns configured BaselineModel with correct defaults and overrides."""
    model = get_baseline_model("logistic_regression", C=0.5)
    assert isinstance(model, BaselineModel)
    assert model.model_name == "logistic_regression"
    assert model._effective_params["C"] == 0.5
    assert model._effective_params["solver"] == "lbfgs"

    # Test invalid model name
    with pytest.raises(ValueError, match="Unsupported model name"):
        get_baseline_model("non_existent_algorithm")


def test_get_all_baseline_models():
    """Verify get_all_baseline_models returns an instantiated dictionary of all 9 models."""
    models = get_all_baseline_models(random_state=42)
    assert len(models) == 9
    for name in BASELINE_MODEL_NAMES:
        assert name in models
        assert isinstance(models[name], BaselineModel)
        assert models[name].model_name == name


@pytest.mark.parametrize("model_name", BASELINE_MODEL_NAMES)
def test_all_9_models_fit_predict_proba(model_name: str, synthetic_lending_club_data: pd.DataFrame):
    """
    Test fit(), predict_proba(), and predict() across all 9 baseline algorithms
    using synthetic LendingClub data.
    """
    df = synthetic_lending_club_data
    X = df[FEATURE_COLS]
    y = df["loan_status"]

    # Use lightweight hyperparameters for fast test execution
    fast_params = {
        "random_state": 42,
    }
    if model_name in ["random_forest", "adaboost", "gbdt", "xgboost", "lightgbm"]:
        fast_params["n_estimators"] = 10
    elif model_name == "catboost":
        fast_params["iterations"] = 10
        fast_params["verbose"] = 0
    elif model_name == "mlp":
        fast_params["max_iter"] = 20
        fast_params["hidden_layer_sizes"] = (16, 8)
    elif model_name == "decision_tree":
        fast_params["max_depth"] = 4
    elif model_name == "logistic_regression":
        fast_params["max_iter"] = 100

    try:
        model = get_baseline_model(model_name, **fast_params)
    except ImportError as e:
        pytest.skip(f"Skipping {model_name} due to missing optional library: {e}")

    # 1. Test fit
    model.fit(X, y)
    assert model.is_fitted_
    assert np.array_equal(model.classes_, np.array([0, 1]))
    assert model.n_features_in_ == len(FEATURE_COLS)
    assert model.feature_names_in_ == FEATURE_COLS

    # 2. Test predict_proba
    probs = model.predict_proba(X)
    assert isinstance(probs, np.ndarray)
    assert probs.shape == (len(df), 2)
    # Probabilities strictly in [0, 1]
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    # Rows sum to 1.0
    row_sums = probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5)

    # 3. Test predict with default threshold 0.5
    preds = model.predict(X, threshold=0.5)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(df),)
    assert set(np.unique(preds)).issubset({0, 1})
    expected_preds = (probs[:, 1] >= 0.5).astype(int)
    assert np.array_equal(preds, expected_preds)

    # 4. Test predict with custom thresholds
    preds_conservative = model.predict(X, threshold=0.3)
    preds_aggressive = model.predict(X, threshold=0.7)
    # Lower threshold predicts equal or more positive defaults
    assert np.sum(preds_conservative) >= np.sum(preds_aggressive)


def test_model_unfitted_raises():
    """Verify calling predict or predict_proba before fit raises ValueError."""
    model = get_baseline_model("decision_tree")
    dummy_X = np.ones((5, 10))
    with pytest.raises(ValueError, match="is not fitted yet"):
        model.predict_proba(dummy_X)

    with pytest.raises(ValueError, match="is not fitted yet"):
        model.predict(dummy_X)


def test_invalid_threshold_raises(synthetic_lending_club_data: pd.DataFrame):
    """Verify invalid threshold values outside (0, 1) raise ValueError."""
    df = synthetic_lending_club_data
    X = df[FEATURE_COLS]
    y = df["loan_status"]

    model = get_baseline_model("logistic_regression", max_iter=50)
    model.fit(X, y)

    for invalid_tau in [-0.1, 0.0, 1.0, 1.5]:
        with pytest.raises(ValueError, match="Classification threshold must be strictly within"):
            model.predict(X, threshold=invalid_tau)


def test_run_baseline_replication(synthetic_lending_club_data: pd.DataFrame):
    """Verify the baseline replication benchmark runner executes CV and returns all metrics."""
    df = synthetic_lending_club_data
    X = df[FEATURE_COLS].iloc[:100]
    y = df["loan_status"].iloc[:100]

    results = run_baseline_replication(
        X,
        y,
        cv=2,
        models=["decision_tree", "logistic_regression"],
        random_state=42,
    )

    assert "decision_tree" in results
    assert "logistic_regression" in results

    dt_res = results["decision_tree"]
    assert "summary" in dt_res
    assert "folds" in dt_res

    for metric in ["roc_auc", "pr_auc", "brier_score", "accuracy", "f1", "precision", "recall"]:
        assert metric in dt_res["summary"]
        assert "mean" in dt_res["summary"][metric]
        assert 0.0 <= dt_res["summary"][metric]["mean"] <= 1.0


# =====================================================================
# Unit Tests: Calibration Engine (CREDEX-09)
# =====================================================================


def test_compute_brier_score():
    """Verify analytical Brier score matches expected values and sklearn implementation."""
    y_true = np.array([0, 1, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.8, 0.2])

    score = compute_brier_score(y_true, y_prob)
    expected_sklearn = brier_score_loss(y_true, y_prob)
    assert np.isclose(score, expected_sklearn, atol=1e-6)

    # Perfect predictions = 0.0
    assert compute_brier_score(y_true, [0.0, 1.0, 1.0, 0.0]) == 0.0
    # Worst possible predictions = 1.0
    assert compute_brier_score(y_true, [1.0, 0.0, 0.0, 1.0]) == 1.0

    # Test error on mismatched lengths
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_brier_score([0, 1], [0.5])


def test_compute_expected_calibration_error():
    """Verify ECE calculation on calibrated and uncalibrated distributions."""
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.05, 0.1, 0.9, 0.95])

    ece = compute_expected_calibration_error(y_true, y_prob, n_bins=5)
    assert 0.0 <= ece <= 1.0
    assert ece < 0.2  # Well-aligned predictions should have low ECE


def test_compute_calibration_curve():
    """Verify calibration curve computes empirical frequencies and mean probabilities."""
    y_true = np.array([0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.7, 0.8, 0.9])

    prob_true, prob_pred = compute_calibration_curve(y_true, y_prob, n_bins=5)
    assert len(prob_true) == len(prob_pred)
    assert len(prob_true) > 0
    assert np.all((prob_true >= 0.0) & (prob_true <= 1.0))
    assert np.all((prob_pred >= 0.0) & (prob_pred <= 1.0))


@pytest.mark.parametrize("method", ["isotonic", "platt", "sigmoid"])
def test_probability_calibrator_methods(method: str):
    """Verify ProbabilityCalibrator fits and predicts valid probabilities for all methods."""
    np.random.seed(42)
    # Raw uncalibrated probabilities skewed high
    raw_probs = np.random.uniform(0.3, 0.9, size=200)
    # True binary default labels correlated with raw probs
    y_true = (raw_probs + np.random.normal(0, 0.2, size=200) >= 0.6).astype(int)

    calibrator = ProbabilityCalibrator(method=method)
    calibrator.fit(raw_probs, y_true)
    assert calibrator.is_fitted_

    # Test calibrate (1D output)
    test_probs = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    calibrated = calibrator.calibrate(test_probs)
    assert isinstance(calibrated, np.ndarray)
    assert calibrated.shape == test_probs.shape
    assert np.all(calibrated >= 0.0)
    assert np.all(calibrated <= 1.0)

    # Test predict_proba (2D output)
    cal_2d = calibrator.predict_proba(test_probs)
    assert cal_2d.shape == (len(test_probs), 2)
    assert np.allclose(cal_2d.sum(axis=1), 1.0, atol=1e-6)

    # Test predict (binary labels)
    preds = calibrator.predict(test_probs, threshold=0.5)
    assert set(np.unique(preds)).issubset({0, 1})


def test_out_of_fold_calibrator(synthetic_lending_club_data: pd.DataFrame):
    """Verify OutOfFoldCalibrator conducts cross-validation, calibrates, and fits final estimator."""
    df = synthetic_lending_club_data
    X = df[FEATURE_COLS].iloc[:150]
    y = df["loan_status"].iloc[:150]

    base_model = get_baseline_model("logistic_regression", max_iter=100, random_state=42)
    oof_cal = OutOfFoldCalibrator(base_estimator=base_model, method="isotonic", cv=3, random_state=42)

    oof_cal.fit(X, y)
    assert oof_cal.is_fitted_
    assert oof_cal.oof_predictions_ is not None
    assert len(oof_cal.oof_predictions_) == len(y)

    # Predict probabilities
    probs = oof_cal.predict_proba(X)
    assert probs.shape == (len(X), 2)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    # Predict labels
    preds = oof_cal.predict(X, threshold=0.5)
    assert len(preds) == len(X)
    assert set(np.unique(preds)).issubset({0, 1})


def test_evaluate_calibration():
    """Verify evaluate_calibration computes comparative metrics and lifts."""
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    raw_probs = np.array([0.4, 0.45, 0.55, 0.6, 0.35, 0.7, 0.4, 0.65])
    cal_probs = np.array([0.1, 0.15, 0.85, 0.9, 0.05, 0.95, 0.1, 0.85])

    eval_dict = evaluate_calibration(y_true, raw_probs, cal_probs)
    assert "brier_score" in eval_dict
    assert "expected_calibration_error" in eval_dict
    assert "log_loss" in eval_dict
    assert "calibration_curves" in eval_dict

    assert eval_dict["brier_score"]["calibrated"] < eval_dict["brier_score"]["uncalibrated"]
    assert eval_dict["brier_score"]["improvement_pct"] > 0.0


# =====================================================================
# Hypothesis Property-Based Tests
# =====================================================================


@settings(max_examples=50, deadline=None)
@given(
    uncalibrated_probs=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=20,
        max_size=100,
    ),
    binary_targets=st.lists(
        st.integers(min_value=0, max_value=1),
        min_size=20,
        max_size=100,
    ),
    test_inputs=st.lists(
        st.floats(min_value=-0.5, max_value=1.5, allow_nan=False, allow_infinity=False),
        min_size=5,
        max_size=50,
    ),
)
def test_property_calibrated_probabilities_strictly_bounded(
    uncalibrated_probs: List[float],
    binary_targets: List[int],
    test_inputs: List[float],
):
    """
    Property Test: Calibrated probabilities remain strictly bounded within [0, 1]
    regardless of input distributions, extreme values, or out-of-bound inputs.
    """
    # Align training length
    n = min(len(uncalibrated_probs), len(binary_targets))
    y_p = np.array(uncalibrated_probs[:n])
    y_t = np.array(binary_targets[:n])

    # Ensure both classes are present for logistic fit
    if len(np.unique(y_t)) < 2:
        y_t[0] = 0
        y_t[1] = 1

    for method in ["isotonic", "platt"]:
        calibrator = ProbabilityCalibrator(method=method)
        calibrator.fit(y_p, y_t)

        test_arr = np.array(test_inputs)
        calibrated = calibrator.calibrate(test_arr)

        # 1. No NaNs or Infs
        assert not np.isnan(calibrated).any(), f"NaNs found in {method} calibration"
        assert not np.isinf(calibrated).any(), f"Infs found in {method} calibration"

        # 2. Strict [0, 1] bounds
        assert np.all(calibrated >= 0.0), f"Negative probability in {method}: {calibrated.min()}"
        assert np.all(calibrated <= 1.0), f"Probability > 1 in {method}: {calibrated.max()}"

        # 3. 2D probabilities sum to 1.0
        cal_2d = calibrator.predict_proba(test_arr)
        assert np.allclose(cal_2d.sum(axis=1), 1.0, atol=1e-5)


@settings(max_examples=30, deadline=None)
@given(
    probs=st.lists(
        st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False),
        min_size=20,
        max_size=60,
    ),
    targets=st.lists(
        st.integers(min_value=0, max_value=1),
        min_size=20,
        max_size=60,
    ),
)
def test_property_isotonic_monotonicity(probs: List[float], targets: List[int]):
    """
    Property Test: Isotonic regression produces a monotonically non-decreasing calibration curve.
    If p_1 <= p_2, then calibrated(p_1) <= calibrated(p_2).
    """
    n = min(len(probs), len(targets))
    y_p = np.array(probs[:n])
    y_t = np.array(targets[:n])

    calibrator = ProbabilityCalibrator(method="isotonic")
    calibrator.fit(y_p, y_t)

    # Sorted evaluation probe
    sorted_probe = np.linspace(0.0, 1.0, 50)
    calibrated_probe = calibrator.calibrate(sorted_probe)

    # Check non-decreasing monotonicity: diff >= 0
    diffs = np.diff(calibrated_probe)
    assert np.all(diffs >= -1e-9), f"Monotonicity violated in Isotonic Calibration: {diffs.min()}"


@settings(max_examples=50, deadline=None)
@given(
    y_true=st.lists(
        st.integers(min_value=0, max_value=1),
        min_size=5,
        max_size=50,
    ),
    y_prob=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=5,
        max_size=50,
    ),
)
def test_property_brier_score_bounds(y_true: List[int], y_prob: List[float]):
    """
    Property Test: Brier score is non-negative and strictly <= 1.0 for any binary targets
    and probabilities in [0, 1].
    """
    n = min(len(y_true), len(y_prob))
    yt = np.array(y_true[:n])
    yp = np.array(y_prob[:n])

    score = compute_brier_score(yt, yp)
    assert 0.0 <= score <= 1.0
    assert not np.isnan(score)
    assert not np.isinf(score)
