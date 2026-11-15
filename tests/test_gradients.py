"""
Unit and Property-Based Tests for Analytical Custom Loss Gradients (CREDEX-10).

Verifies:
1. Analytical first-order gradients match numerical finite-difference approximations
   (|g_analytical - g_numerical| < 1e-5).
2. Analytical second-order Hessians match numerical finite-difference approximations.
3. For w=1, gradients exactly match standard binary cross-entropy loss derivatives
   (g_i = p_i - y_i).
4. Second-order Hessians are strictly positive (h_i > 0) across all inputs,
   preventing optimizer divergence.
5. Numerical stability under extreme margin values (e.g., logits in [-100, 100])
   without NaNs or Infs.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.models.objective_functions import (
    cost_sensitive_gradients,
    cost_sensitive_loss,
    cost_sensitive_loss_elementwise,
    numerical_gradient,
    numerical_hessian,
    sigmoid,
)

# ==============================================================================
# Deterministic Unit Tests
# ==============================================================================


def test_sigmoid_numerical_stability():
    """Verify sigmoid handles extreme values without overflow or NaN."""
    extreme_z = np.array([-1000.0, -100.0, -10.0, 0.0, 10.0, 100.0, 1000.0])
    probs = sigmoid(extreme_z)

    assert not np.isnan(probs).any()
    assert not np.isinf(probs).any()
    assert probs[0] == pytest.approx(0.0, abs=1e-12)
    assert probs[3] == pytest.approx(0.5, abs=1e-12)
    assert probs[-1] == pytest.approx(1.0, abs=1e-12)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_analytical_gradient_vs_numerical_finite_difference():
    """
    Acceptance Criteria: Analytical gradients match numerical finite-difference
    approximations within tolerance |g_analytical - g_numerical| < 1e-5.
    """
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    margins = np.array([-3.0, -1.5, -0.5, 0.0, 0.5, 1.2, 2.5, 4.0])

    for w in [1.0, 5.0, 10.0, 20.0]:
        g_analytical, h_analytical = cost_sensitive_gradients(y_true, margins, w=w, epsilon=1e-6)
        g_numerical = numerical_gradient(y_true, margins, w=w, delta=1e-6)

        max_grad_diff = np.max(np.abs(g_analytical - g_numerical))
        assert (
            max_grad_diff < 1e-5
        ), f"Gradient mismatch for w={w}: max difference = {max_grad_diff} >= 1e-5"


def test_analytical_hessian_vs_numerical_finite_difference():
    """Verify analytical second-order Hessian matches finite-difference approximation."""
    y_true = np.array([1, 0, 1, 0, 1, 0])
    margins = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])

    for w in [1.0, 5.0, 10.0]:
        _, h_analytical = cost_sensitive_gradients(y_true, margins, w=w, epsilon=1e-6)
        h_numerical = numerical_hessian(y_true, margins, w=w, delta=1e-6, epsilon=1e-6)

        max_hess_diff = np.max(np.abs(h_analytical - h_numerical))
        assert (
            max_hess_diff < 1e-4
        ), f"Hessian mismatch for w={w}: max difference = {max_hess_diff} >= 1e-4"


def test_standard_cross_entropy_equivalence_when_w_is_one():
    """
    Acceptance Criteria: For w = 1, gradients exactly match standard binary
    cross-entropy loss derivatives:
        g_i = p_i - y_i
        h_i = p_i * (1 - p_i) + epsilon
    """
    y_true = np.array([1, 0, 1, 0, 0, 1])
    margins = np.array([-2.5, -1.0, 0.0, 0.8, 1.5, 3.0])

    probs = sigmoid(margins)
    expected_grad = probs - y_true
    expected_hess = probs * (1.0 - probs) + 1e-6

    g_analytical, h_analytical = cost_sensitive_gradients(y_true, margins, w=1.0, epsilon=1e-6)

    np.testing.assert_allclose(g_analytical, expected_grad, atol=1e-12)
    np.testing.assert_allclose(h_analytical, expected_hess, atol=1e-12)


def test_hessian_strict_positivity():
    """
    Acceptance Criteria: Second-order Hessians are strictly positive (h_i > 0)
    across all inputs, preventing tree optimizer divergence.
    """
    margins = np.linspace(-50.0, 50.0, num=500)
    y_ones = np.ones_like(margins)
    y_zeros = np.zeros_like(margins)

    epsilon = 1e-6
    for y_true in [y_ones, y_zeros]:
        for w in [1.0, 5.0, 10.0, 25.0]:
            _, hess = cost_sensitive_gradients(y_true, margins, w=w, epsilon=epsilon)
            assert np.all(hess > 0.0), f"Hessian non-positive for w={w}"
            assert np.all(hess >= epsilon), f"Hessian below damping threshold {epsilon}"


def test_mismatched_dimensions_raise_error():
    """Verify ValueError is raised when labels and margins have different shapes."""
    y_true = np.array([1, 0, 1])
    margins = np.array([0.1, -0.2])

    with pytest.raises(ValueError, match="Shape mismatch"):
        cost_sensitive_gradients(y_true, margins)

    with pytest.raises(ValueError, match="Shape mismatch"):
        cost_sensitive_loss(y_true, np.array([0.5, 0.5]))


def test_cost_sensitive_loss_reductions():
    """Verify 'mean', 'sum', and 'none' reduction modes for cost_sensitive_loss."""
    y_true = np.array([1, 0])
    probs = np.array([0.8, 0.2])
    w = 5.0

    loss_none = cost_sensitive_loss(y_true, probs, w=w, reduction="none")
    assert len(loss_none) == 2
    assert loss_none[0] == pytest.approx(-5.0 * np.log(0.8), rel=1e-6)
    assert loss_none[1] == pytest.approx(-1.0 * np.log(0.8), rel=1e-6)

    loss_mean = cost_sensitive_loss(y_true, probs, w=w, reduction="mean")
    assert loss_mean == pytest.approx(float(np.mean(loss_none)), rel=1e-6)

    loss_sum = cost_sensitive_loss(y_true, probs, w=w, reduction="sum")
    assert loss_sum == pytest.approx(float(np.sum(loss_none)), rel=1e-6)

    with pytest.raises(ValueError, match="Unknown reduction"):
        cost_sensitive_loss(y_true, probs, reduction="invalid")


# ==============================================================================
# Hypothesis Property-Based Tests
# ==============================================================================


@settings(max_examples=100, deadline=None)
@given(
    y=st.integers(min_value=0, max_value=1),
    margin=st.floats(min_value=-15.0, max_value=15.0),
    w=st.floats(min_value=1.0, max_value=50.0),
)
def test_hypothesis_gradient_matches_finite_difference(y, margin, w):
    """Property test: Analytical gradients match numerical finite-difference approximation."""
    y_arr = np.array([y], dtype=np.float64)
    z_arr = np.array([margin], dtype=np.float64)

    g_ana, _ = cost_sensitive_gradients(y_arr, z_arr, w=w, epsilon=1e-6)
    g_num = numerical_gradient(y_arr, z_arr, w=w, delta=1e-6)

    assert abs(g_ana[0] - g_num[0]) < 1e-5


@settings(max_examples=100, deadline=None)
@given(
    y=st.integers(min_value=0, max_value=1),
    margin=st.floats(min_value=-30.0, max_value=30.0),
)
def test_hypothesis_w1_matches_standard_logloss(y, margin):
    """Property test: For w=1, analytical derivatives match standard logloss."""
    y_arr = np.array([y], dtype=np.float64)
    z_arr = np.array([margin], dtype=np.float64)

    p = sigmoid(z_arr)
    expected_g = p - y_arr
    expected_h = p * (1.0 - p) + 1e-6

    g_ana, h_ana = cost_sensitive_gradients(y_arr, z_arr, w=1.0, epsilon=1e-6)

    assert abs(g_ana[0] - expected_g[0]) < 1e-10
    assert abs(h_ana[0] - expected_h[0]) < 1e-10


@settings(max_examples=100, deadline=None)
@given(
    y=st.integers(min_value=0, max_value=1),
    margin=st.floats(min_value=-50.0, max_value=50.0),
    w=st.floats(min_value=1.0, max_value=100.0),
)
def test_hypothesis_hessian_strictly_positive(y, margin, w):
    """Property test: Hessian is strictly positive (h_i >= epsilon > 0) across all inputs."""
    y_arr = np.array([y], dtype=np.float64)
    z_arr = np.array([margin], dtype=np.float64)
    epsilon = 1e-6

    _, h_ana = cost_sensitive_gradients(y_arr, z_arr, w=w, epsilon=epsilon)

    assert h_ana[0] >= epsilon
    assert not np.isnan(h_ana[0])
    assert not np.isinf(h_ana[0])


@settings(max_examples=100, deadline=None)
@given(
    margin=st.floats(min_value=-10.0, max_value=10.0),
    w=st.floats(min_value=1.0, max_value=20.0),
)
def test_hypothesis_loss_non_negative(margin, w):
    """Property test: Elementwise cost-sensitive loss is strictly non-negative."""
    for y in [0, 1]:
        loss = cost_sensitive_loss_elementwise(y, margin, w=w)
        assert loss >= 0.0
        assert not np.isnan(loss)
        assert not np.isinf(loss)
