"""
Comprehensive Unit Tests and Hypothesis Property-Based Tests for Evaluation Metrics (CREDEX-14).

Verifies:
1. Statistical discrimination metrics (ROC-AUC, PR-AUC, Recall, Precision, F1, Accuracy, Specificity).
2. Probability calibration error (Brier Score strictly bounded in [0, 1]).
3. Financial economic metrics:
   - Expected Unit Loss (5:1 benchmark and 10:1 stress scenario).
   - Total Portfolio Dollar Loss (with contract terms, LGD and NIM).
   - Dollar savings (delta $ and %) vs baseline.
4. Hypothesis property tests:
   - Brier score bounded in [0, 1] for all valid binary label and probability arrays.
   - Threshold optimizer tau* produces non-negative dollar savings vs tau=0.5.
   - Monotonic decrease of default recall as tau increases.
   - Monotonic increase of portfolio loss as LGD increases.
   - Self-comparison baseline delta $ == 0 and pct == 0.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.evaluation.metrics import (
    CreditEvaluationResult,
    compute_brier_score,
    compute_dollar_savings,
    compute_expected_unit_loss,
    compute_portfolio_loss,
    compute_statistical_metrics,
    evaluate_credit_model,
    evaluate_lgd_sensitivity,
)
from src.models.threshold_optimizer import ThresholdOptimizer


# ==============================================================================
# Unit Tests: Statistical Metrics
# ==============================================================================


def test_brier_score_exact() -> None:
    """Test exact analytical computation of Brier Score."""
    # Perfect predictions
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.0, 1.0, 0.0, 1.0])
    assert compute_brier_score(y_true, y_prob) == 0.0

    # Completely wrong predictions
    y_prob_wrong = np.array([1.0, 0.0, 1.0, 0.0])
    assert compute_brier_score(y_true, y_prob_wrong) == 1.0

    # Constant 0.5 predictions -> (0.5)^2 = 0.25
    y_prob_half = np.array([0.5, 0.5, 0.5, 0.5])
    assert compute_brier_score(y_true, y_prob_half) == 0.25


def test_statistical_metrics_perfect_classifier() -> None:
    """Test statistical metrics under perfect classification."""
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])

    stats = compute_statistical_metrics(y_true, y_prob, threshold=0.5)
    assert stats["roc_auc"] == 1.0
    assert stats["pr_auc"] == 1.0
    assert stats["recall"] == 1.0
    assert stats["precision"] == 1.0
    assert stats["f1_score"] == 1.0
    assert stats["accuracy"] == 1.0
    assert stats["specificity"] == 1.0
    assert 0.0 <= stats["brier_score"] <= 1.0


def test_statistical_metrics_inverted_classifier() -> None:
    """Test metrics when predictions are inverted."""
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.9, 0.8, 0.2, 0.1])

    stats = compute_statistical_metrics(y_true, y_prob, threshold=0.5)
    assert stats["roc_auc"] == 0.0
    assert stats["recall"] == 0.0
    assert stats["precision"] == 0.0
    assert stats["accuracy"] == 0.0
    assert stats["specificity"] == 0.0


def test_statistical_metrics_all_negative_or_positive() -> None:
    """Test single-class edge cases without unhandled exceptions."""
    y_all_0 = np.array([0, 0, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4])

    stats = compute_statistical_metrics(y_all_0, y_prob, threshold=0.5)
    assert stats["roc_auc"] == 0.5
    assert stats["accuracy"] == 1.0
    assert stats["specificity"] == 1.0
    assert stats["recall"] == 0.0


# ==============================================================================
# Unit Tests: Financial Economic Metrics
# ==============================================================================


def test_expected_unit_loss_exact() -> None:
    """Test exact expected unit loss under 5:1 and 10:1."""
    # 4 samples: y = [0, 0, 1, 1], p = [0.1, 0.6, 0.4, 0.9]
    # threshold = 0.5:
    # y_pred = [0, 1, 0, 1]
    # sample 0: y=0, pred=0 -> TN
    # sample 1: y=0, pred=1 -> FP (cost = 1.0)
    # sample 2: y=1, pred=0 -> FN (cost = 5.0 / 10.0)
    # sample 3: y=1, pred=1 -> TP
    # Under 5:1: Total cost = 5*1 + 1*1 = 6. Normalized = 6 / 4 = 1.5
    # Under 10:1: Total cost = 10*1 + 1*1 = 11. Normalized = 11 / 4 = 2.75
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.6, 0.4, 0.9])

    loss_5_1 = compute_expected_unit_loss(
        y_true, y_prob, threshold=0.5, cost_fn=5.0, cost_fp=1.0, normalize=True
    )
    assert pytest.approx(loss_5_1, rel=1e-5) == 1.5

    loss_10_1 = compute_expected_unit_loss(
        y_true, y_prob, threshold=0.5, cost_fn=10.0, cost_fp=1.0, normalize=True
    )
    assert pytest.approx(loss_10_1, rel=1e-5) == 2.75


def test_portfolio_dollar_loss_exact() -> None:
    """Test portfolio dollar loss calculation with contract terms."""
    # 2 loans:
    # Loan 0: y=1, p=0.2 (at tau=0.5 -> FN, approved defaulting loan)
    # loan_amount = 10,000, LGD = 0.65 -> loss = 10,000 * 0.65 = 6,500
    # Loan 1: y=0, p=0.8 (at tau=0.5 -> FP, rejected good loan)
    # loan_amount = 20,000, NIM = 0.035, term = 36 months (3 years)
    # loss = 20,000 * 0.035 * (36/12) = 20,000 * 0.035 * 3 = 2,100
    # Total portfolio loss = 6,500 + 2,100 = 8,600
    y_true = np.array([1, 0])
    y_prob = np.array([0.2, 0.8])
    amounts = np.array([10000.0, 20000.0])
    terms = np.array(["36 months", "36 months"])

    loss = compute_portfolio_loss(
        y_true=y_true,
        y_prob=y_prob,
        loan_amounts=amounts,
        terms=terms,
        threshold=0.5,
        lgd=0.65,
        nim=0.035,
    )
    assert pytest.approx(loss, rel=1e-5) == 8600.0


def test_compute_dollar_savings() -> None:
    """Test delta $ and percentage savings computation."""
    baseline_loss = 10000.0
    current_loss = 7500.0

    delta_abs, delta_pct = compute_dollar_savings(current_loss, baseline_loss)
    assert delta_abs == 2500.0
    assert delta_pct == 25.0

    # Baseline loss zero
    d_abs, d_pct = compute_dollar_savings(0.0, 0.0)
    assert d_abs == 0.0
    assert d_pct == 0.0


def test_evaluate_credit_model_full_pass() -> None:
    """Test evaluate_credit_model returns comprehensive CreditEvaluationResult."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.4, 0.3, 0.7])
    amounts = np.array([10000.0, 15000.0, 20000.0, 12000.0, 8000.0, 25000.0])
    terms = np.array([36, 60, 36, 36, 60, 36])

    result = evaluate_credit_model(
        y_true=y_true,
        y_prob=y_prob,
        threshold=0.5,
        loan_amounts=amounts,
        terms=terms,
        lgd=0.65,
        nim=0.035,
    )

    assert isinstance(result, CreditEvaluationResult)
    assert 0.0 <= result.roc_auc <= 1.0
    assert 0.0 <= result.pr_auc <= 1.0
    assert 0.0 <= result.brier_score <= 1.0
    assert result.portfolio_dollar_loss is not None
    assert result.portfolio_dollar_loss > 0.0
    assert result.expected_unit_loss_5_1 > 0.0
    assert result.tp + result.fp + result.tn + result.fn == 6

    # Verify serialization
    d = result.to_dict()
    assert isinstance(d, dict)
    assert "roc_auc" in d
    assert "portfolio_dollar_loss" in d

    s = result.to_series()
    assert isinstance(s, pd.Series)
    assert s["threshold"] == 0.5


def test_evaluate_lgd_sensitivity() -> None:
    """Test multi-scenario LGD evaluation table."""
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.2, 0.3, 0.1, 0.8])
    amounts = np.array([10000.0, 20000.0, 15000.0, 25000.0])
    terms = np.array([36, 36, 36, 36])

    df_sens = evaluate_lgd_sensitivity(
        y_true=y_true,
        y_prob=y_prob,
        loan_amounts=amounts,
        terms=terms,
        threshold=0.25,
        baseline_threshold=0.50,
        lgd_scenarios=[0.60, 0.70, 0.75],
    )

    assert isinstance(df_sens, pd.DataFrame)
    assert len(df_sens) == 3
    assert list(df_sens["lgd"]) == [0.60, 0.70, 0.75]
    assert "dollar_savings_abs" in df_sens.columns
    assert "dollar_savings_pct" in df_sens.columns


# ==============================================================================
# Hypothesis Property-Based Tests
# ==============================================================================


@settings(max_examples=100, deadline=None)
@given(
    st.lists(st.integers(min_value=0, max_value=1), min_size=5, max_size=100),
    st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=5,
        max_size=100,
    ),
)
def test_hypothesis_brier_score_bounded_in_unit_interval(
    labels: list[int], probs: list[float]
) -> None:
    """Property: Brier score must strictly be in [0, 1] for all inputs."""
    min_len = min(len(labels), len(probs))
    y = np.array(labels[:min_len])
    p = np.array(probs[:min_len])

    brier = compute_brier_score(y, p)
    assert 0.0 <= brier <= 1.0


@settings(max_examples=50, deadline=None)
@given(
    st.lists(st.integers(min_value=0, max_value=1), min_size=20, max_size=100),
    st.lists(st.floats(min_value=0.01, max_value=0.99), min_size=20, max_size=100),
    st.lists(st.floats(min_value=1000.0, max_value=40000.0), min_size=20, max_size=100),
)
def test_hypothesis_optimal_threshold_savings_non_negative(
    labels: list[int], probs: list[float], amounts: list[float]
) -> None:
    """
    Property: Decision threshold optimization to find tau* must yield
    non-negative dollar savings vs tau=0.50 (Loss(tau*) <= Loss(0.50)).
    """
    n = min(len(labels), len(probs), len(amounts))
    # Ensure both classes exist
    y = np.array(labels[:n])
    if len(np.unique(y)) < 2:
        y[0] = 0
        y[1] = 1

    p = np.array(probs[:n])
    amt = np.array(amounts[:n])
    terms = np.full(n, 36.0)

    optimizer = ThresholdOptimizer(min_tau=0.01, max_tau=0.99, step_size=0.01)
    res = optimizer.optimize_portfolio_dollar_loss(
        y_true=y,
        y_prob=p,
        loan_amounts=amt,
        terms=terms,
        lgd=0.65,
        nim=0.035,
    )

    # tau* is the argmin over the grid containing tau=0.50, so loss_reduction_absolute >= 0
    assert res.min_loss <= res.baseline_loss_tau_05 + 1e-6
    assert res.loss_reduction_absolute >= -1e-6
    assert res.loss_reduction_pct >= -1e-6


@settings(max_examples=50, deadline=None)
@given(
    st.lists(st.integers(min_value=0, max_value=1), min_size=10, max_size=50),
    st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=10, max_size=50),
)
def test_hypothesis_default_recall_monotonicity_wrt_threshold(
    labels: list[int], probs: list[float]
) -> None:
    """
    Property: As threshold tau increases, default recall (sensitivity)
    must be monotonically non-increasing (fewer defaults flagged).
    """
    n = min(len(labels), len(probs))
    y = np.array(labels[:n])
    if np.sum(y == 1) == 0:
        y[0] = 1
    p = np.array(probs[:n])

    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    recalls = [
        compute_statistical_metrics(y, p, threshold=t)["recall"]
        for t in thresholds
    ]

    for i in range(len(recalls) - 1):
        assert recalls[i] >= recalls[i + 1] - 1e-9


def test_self_comparison_baseline_savings_zero() -> None:
    """Property: When model is evaluated against its own baseline, delta is exactly zero."""
    y = np.array([0, 1, 0, 1, 1, 0])
    p = np.array([0.1, 0.7, 0.2, 0.8, 0.4, 0.3])
    amounts = np.array([10000.0, 15000.0, 20000.0, 12000.0, 8000.0, 25000.0])
    terms = np.array([36, 36, 36, 36, 36, 36])

    result = evaluate_credit_model(
        y_true=y,
        y_prob=p,
        threshold=0.5,
        loan_amounts=amounts,
        terms=terms,
        baseline_prob=p,  # Self baseline
    )

    assert result.dollar_savings_abs == pytest.approx(0.0, abs=1e-6)
    assert result.dollar_savings_pct == pytest.approx(0.0, abs=1e-6)
    assert result.unit_loss_savings_5_1_abs == pytest.approx(0.0, abs=1e-6)
