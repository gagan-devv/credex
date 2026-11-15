"""
Unit and Property-Based Tests for Economic Threshold Optimizer (CREDEX-12).

Verifies:
1. Optimal tau* under C_FN:C_FP = 5:1 is strictly less than 0.50 (typically tau* in [0.15, 0.30]).
2. Property test: As w = C_FN / C_FP increases, optimal tau* monotonically decreases
   (higher risk aversion).
3. Losses are strictly non-negative across all candidate thresholds.
4. Portfolio dollar loss correctly computes LGD * loan_amount vs NIM * term/12 * loan_amount.
5. Robust handling of term formats (strings '36 months', '60 months', and numeric).
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.models.threshold_optimizer import (
    ThresholdOptimizationResult,
    ThresholdOptimizer,
    compute_confusion_at_thresholds,
    compute_portfolio_dollar_loss,
    compute_unit_expected_loss,
    parse_term_in_months,
)

# ==============================================================================
# Deterministic Unit Tests
# ==============================================================================


def test_parse_term_in_months():
    """Verify parsing of various loan term formats."""
    terms_str = ["36 months", "60 months", " 36 months ", "60"]
    parsed = parse_term_in_months(terms_str)
    np.testing.assert_allclose(parsed, [36.0, 60.0, 36.0, 60.0])

    terms_num = np.array([36, 60, 36])
    parsed_num = parse_term_in_months(terms_num)
    np.testing.assert_allclose(parsed_num, [36.0, 60.0, 36.0])

    terms_series = pd.Series(["36 months", None, "60 months"])
    parsed_series = parse_term_in_months(terms_series)
    np.testing.assert_allclose(parsed_series, [36.0, 36.0, 60.0])


def test_confusion_at_thresholds_exactness():
    """Verify confusion matrix counts match manual calculation across thresholds."""
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.2, 0.8, 0.3, 0.7])
    thresholds = np.array([0.1, 0.25, 0.5, 0.75, 0.9])

    conf = compute_confusion_at_thresholds(y_true, y_prob, thresholds)

    # At tau = 0.5:
    # y_prob >= 0.5: indices 1 (y=1) -> TP, and 3 (y=0) -> FP
    # y_prob < 0.5: indices 0 (y=1) -> FN, and 2 (y=0) -> TN
    idx_05 = 2
    assert conf["tp"][idx_05] == 1
    assert conf["fp"][idx_05] == 1
    assert conf["fn"][idx_05] == 1
    assert conf["tn"][idx_05] == 1

    # Total must equal N at every threshold
    total_samples = conf["tp"] + conf["fp"] + conf["fn"] + conf["tn"]
    assert np.all(total_samples == 4)


def test_optimal_tau_less_than_05_for_5_to_1():
    """
    Acceptance Criteria: Optimal tau* under C_FN:C_FP = 5:1 is strictly less than 0.50
    (typically tau* in [0.15, 0.30]).
    """
    np.random.seed(42)
    n = 2000
    # True probabilities with mean ~ 0.20 (typical retail banking default rate)
    true_probs = np.random.beta(a=2, b=8, size=n)
    y_true = np.random.binomial(n=1, p=true_probs)

    optimizer = ThresholdOptimizer(min_tau=0.01, max_tau=0.99, step_size=0.005)
    res = optimizer.optimize_unit_loss(y_true, true_probs, cost_fn=5.0, cost_fp=1.0)

    assert res.optimal_tau < 0.50, f"Expected tau* < 0.50, got {res.optimal_tau}"
    # Theoretical Bayes threshold for w=5 is 1 / (1 + 5) = 0.167
    assert 0.10 <= res.optimal_tau <= 0.35, f"Expected tau* in [0.10, 0.35], got {res.optimal_tau}"
    assert (
        res.min_loss < res.baseline_loss_tau_05
    ), "Optimal loss must be lower than baseline at 0.50"
    assert res.loss_reduction_pct > 0.0


def test_monotonic_decrease_of_tau_with_increasing_penalty():
    """
    Acceptance Criteria: As w = C_FN / C_FP increases, optimal tau* strictly monotonically
    decreases (higher risk aversion).
    """
    np.random.seed(42)
    n = 3000
    true_probs = np.random.beta(a=2, b=6, size=n)
    y_true = np.random.binomial(n=1, p=true_probs)

    optimizer = ThresholdOptimizer(min_tau=0.01, max_tau=0.99, step_size=0.005)

    weights = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    optimal_taus = []

    for w in weights:
        res = optimizer.optimize_unit_loss(y_true, true_probs, cost_fn=w, cost_fp=1.0)
        optimal_taus.append(res.optimal_tau)

    # Verify non-increasing sequence (monotonically decreasing)
    for i in range(len(optimal_taus) - 1):
        assert optimal_taus[i] >= optimal_taus[i + 1], (
            f"Monotonicity violated: w={weights[i]} -> tau*={optimal_taus[i]} "
            f"vs w={weights[i+1]} -> tau*={optimal_taus[i+1]}"
        )

    # Strict decrease between endpoints
    assert optimal_taus[0] > optimal_taus[-1]


def test_portfolio_dollar_loss_exactness():
    """
    Acceptance Criteria: Total Portfolio Dollar Loss =
        sum_{i in FN} (loan_amount_i * LGD) + sum_{j in FP} (loan_amount_j * NIM * term_j / 12).
    """
    y_true = np.array([1, 0])
    y_prob = np.array([0.2, 0.8])
    loan_amounts = np.array([10000.0, 20000.0])
    terms = np.array(["36 months", "60 months"])
    lgd = 0.70
    nim = 0.035

    # At tau = 0.50:
    # Sample 0 (y=1, p=0.2 < 0.5): FN -> Loss = 10000 * 0.70 = $7000
    # Sample 1 (y=0, p=0.8 >= 0.5): FP ->
    # Loss = 20000 * 0.035 * (60 / 12) = 20000 * 0.035 * 5 = $3500
    # Total expected loss = $10,500
    thresholds = np.array([0.50])
    dollar_loss, fn_counts, fp_counts = compute_portfolio_dollar_loss(
        y_true, y_prob, loan_amounts, terms, thresholds, lgd=lgd, nim=nim
    )

    assert dollar_loss[0] == pytest.approx(10500.0, rel=1e-6)
    assert fn_counts[0] == 1
    assert fp_counts[0] == 1


def test_portfolio_dollar_loss_optimizer():
    """Verify ThresholdOptimizer.optimize_portfolio_dollar_loss returns valid results."""
    np.random.seed(42)
    n = 1000
    probs = np.random.uniform(0.05, 0.60, size=n)
    y_true = np.random.binomial(n=1, p=probs)
    loan_amounts = np.random.uniform(5000, 35000, size=n)
    terms = np.random.choice(["36 months", "60 months"], size=n)

    optimizer = ThresholdOptimizer(min_tau=0.01, max_tau=0.99, step_size=0.005)
    res = optimizer.optimize_portfolio_dollar_loss(
        y_true, probs, loan_amounts, terms, lgd=0.65, nim=0.035
    )

    assert isinstance(res, ThresholdOptimizationResult)
    assert res.min_loss >= 0.0
    assert res.baseline_loss_tau_05 >= res.min_loss
    assert res.loss_reduction_absolute >= 0.0
    assert res.loss_type == "portfolio_dollar"

    summary_dict = res.to_dict()
    assert "optimal_tau" in summary_dict
    assert "min_loss" in summary_dict

    df_sweep = res.to_dataframe()
    assert len(df_sweep) == len(optimizer.thresholds)
    assert "threshold" in df_sweep.columns
    assert "loss" in df_sweep.columns


def test_non_negative_losses_across_entire_sweep():
    """
    Acceptance Criteria: Losses are non-negative across all candidate thresholds.
    """
    np.random.seed(123)
    n = 200
    probs = np.random.uniform(0.0, 1.0, size=n)
    y_true = np.random.binomial(n=1, p=probs)
    amounts = np.random.uniform(1000, 40000, size=n)
    terms = np.random.choice([36, 60], size=n)

    thresholds = np.linspace(0.01, 0.99, 50)
    unit_losses, _, _ = compute_unit_expected_loss(
        y_true, probs, thresholds, cost_fn=5.0, cost_fp=1.0
    )
    assert np.all(unit_losses >= 0.0)

    dollar_losses, _, _ = compute_portfolio_dollar_loss(
        y_true, probs, amounts, terms, thresholds, lgd=0.70, nim=0.035
    )
    assert np.all(dollar_losses >= 0.0)


# ==============================================================================
# Hypothesis Property-Based Tests
# ==============================================================================


@settings(max_examples=50, deadline=None)
@given(
    cost_fn=st.floats(min_value=1.0, max_value=50.0),
    cost_fp=st.floats(min_value=0.5, max_value=5.0),
)
def test_hypothesis_unit_loss_non_negative(cost_fn, cost_fp):
    """Property test: Unit expected loss is always non-negative for any positive cost parameters."""
    np.random.seed(42)
    y_true = np.random.randint(0, 2, size=50)
    y_prob = np.random.uniform(0.0, 1.0, size=50)
    thresholds = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

    losses, _, _ = compute_unit_expected_loss(
        y_true, y_prob, thresholds, cost_fn=cost_fn, cost_fp=cost_fp
    )

    assert np.all(losses >= 0.0)
    assert not np.isnan(losses).any()


@settings(max_examples=30, deadline=None)
@given(
    w1=st.floats(min_value=1.0, max_value=5.0),
    w2=st.floats(min_value=10.0, max_value=30.0),
)
def test_hypothesis_monotonic_risk_aversion(w1, w2):
    """Property test: Higher w strictly reduces or maintains optimal tau*."""
    np.random.seed(99)
    y_prob = np.random.beta(a=2, b=5, size=500)
    y_true = np.random.binomial(n=1, p=y_prob)

    optimizer = ThresholdOptimizer(min_tau=0.01, max_tau=0.99, step_size=0.01)
    res_w1 = optimizer.optimize_unit_loss(y_true, y_prob, cost_fn=w1, cost_fp=1.0)
    res_w2 = optimizer.optimize_unit_loss(y_true, y_prob, cost_fn=w2, cost_fp=1.0)

    # When FN penalty is higher, bank rejects more aggressively -> lower decision threshold
    assert res_w2.optimal_tau <= res_w1.optimal_tau
