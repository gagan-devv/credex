"""
Asymmetric Cost-Sensitive Binary Cross-Entropy Loss and Analytical Derivatives.

This module implements the custom objective functions, first-order gradients (g_i),
and damped second-order Hessians (h_i) for asymmetric cost-sensitive credit default prediction:

    L_cost(y, p) = - [w * y * log(p) + (1 - y) * log(1 - p)]

where:
    y in {0, 1} is the true binary label (1 = default, 0 = fully paid)
    p = sigma(y_hat) = 1 / (1 + exp(-y_hat)) is the predicted default probability
    y_hat in R is the raw tree margin (logit)
    w = C_FN / C_FP >= 5 is the asymmetric cost penalty multiplier for false negatives

First- and Second-Order Loss Derivatives with respect to raw margin y_hat:
    g_i = dL/dy_hat = p_i * (1 + (w - 1) * y_i) - w * y_i
    h_i = d^2L/dy_hat^2 = p_i * (1 - p_i) * (1 + (w - 1) * y_i) + epsilon

where epsilon = 1e-6 is the Hessian damping factor ensuring strict positivity (h_i > 0)
across all inputs to prevent tree optimizer divergence.
"""

from typing import Tuple, Union

import numpy as np

try:
    from scipy.special import expit as _scipy_expit
except ImportError:
    _scipy_expit = None


def sigmoid(z: Union[float, np.ndarray]) -> np.ndarray:
    """
    Compute numerically stable sigmoid function: sigma(z) = 1 / (1 + exp(-z)).

    Parameters
    ----------
    z : array-like or float
        Raw logit values (margins).

    Returns
    -------
    np.ndarray
        Sigmoid activation values constrained to (0, 1).
    """
    z_arr = np.asarray(z, dtype=np.float64)
    if _scipy_expit is not None:
        return _scipy_expit(z_arr)

    # Numerically stable piecewise sigmoid
    positive_mask = z_arr >= 0
    negative_mask = ~positive_mask

    result = np.empty_like(z_arr, dtype=np.float64)
    result[positive_mask] = 1.0 / (1.0 + np.exp(-z_arr[positive_mask]))

    exp_neg = np.exp(z_arr[negative_mask])
    result[negative_mask] = exp_neg / (1.0 + exp_neg)
    return result


def cost_sensitive_loss_elementwise(
    y_true: Union[float, np.ndarray],
    y_pred_margin: Union[float, np.ndarray],
    w: float = 5.0,
) -> np.ndarray:
    """
    Compute elementwise cost-sensitive binary cross-entropy loss from raw margins.

    Uses log-sum-exp stabilization:
      y = 1: -w * log(sigma(z)) = w * log(1 + exp(-z)) = w * softplus(-z)
      y = 0: -log(1 - sigma(z)) = log(1 + exp(z)) = softplus(z)

    Parameters
    ----------
    y_true : array-like or float
        Ground truth binary labels (0 or 1).
    y_pred_margin : array-like or float
        Predicted raw margins (logits).
    w : float, default=5.0
        Asymmetric false negative penalty multiplier (w = C_FN / C_FP).

    Returns
    -------
    np.ndarray
        Elementwise loss values.
    """
    y = np.asarray(y_true, dtype=np.float64)
    z = np.asarray(y_pred_margin, dtype=np.float64)

    # softplus(x) = max(x, 0) + log(1 + exp(-|x|))
    def _softplus(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))

    loss = w * y * _softplus(-z) + (1.0 - y) * _softplus(z)
    return loss


def cost_sensitive_loss(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    w: float = 5.0,
    eps: float = 1e-15,
    reduction: str = "mean",
) -> float:
    """
    Compute scalar cost-sensitive binary cross-entropy loss from probabilities.

        L = - 1/N sum [ w * y_i * log(p_i) + (1 - y_i) * log(1 - p_i) ]

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_pred_proba : np.ndarray
        Predicted probabilities in [0, 1].
    w : float, default=5.0
        Cost penalty ratio C_FN / C_FP (>= 1.0).
    eps : float, default=1e-15
        Small clipping value to prevent log(0).
    reduction : str, default="mean"
        Reduction method: "mean", "sum", or "none".

    Returns
    -------
    float or np.ndarray
        Computed loss.
    """
    y = np.asarray(y_true, dtype=np.float64).ravel()
    p = np.asarray(y_pred_proba, dtype=np.float64).ravel()

    if len(y) != len(p):
        raise ValueError(f"Shape mismatch: y_true ({y.shape}) vs y_pred_proba ({p.shape})")

    p_clipped = np.clip(p, eps, 1.0 - eps)
    elementwise = -(w * y * np.log(p_clipped) + (1.0 - y) * np.log(1.0 - p_clipped))

    if reduction == "mean":
        return float(np.mean(elementwise))
    elif reduction == "sum":
        return float(np.sum(elementwise))
    elif reduction == "none":
        return elementwise
    else:
        raise ValueError(f"Unknown reduction: {reduction}. Choose 'mean', 'sum', or 'none'.")


def cost_sensitive_gradients(
    y_true: Union[float, np.ndarray],
    y_pred_margin: Union[float, np.ndarray],
    w: float = 5.0,
    epsilon: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute analytical 1st (g_i) and 2nd (h_i) order derivatives with respect to margin y_hat.

        g_i = p_i * (1 + (w - 1) * y_i) - w * y_i
        h_i = p_i * (1 - p_i) * (1 + (w - 1) * y_i) + epsilon

    Parameters
    ----------
    y_true : array-like
        True binary labels (0 or 1).
    y_pred_margin : array-like
        Predicted raw margins (logits).
    w : float, default=5.0
        Cost penalty ratio C_FN / C_FP (w >= 1.0).
    epsilon : float, default=1e-6
        Hessian damping factor ensuring strict positivity (h_i > 0).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (grad, hess):
        - grad (g_i): First-order loss derivatives.
        - hess (h_i): Damped second-order loss derivatives.
    """
    y = np.asarray(y_true, dtype=np.float64).ravel()
    z = np.asarray(y_pred_margin, dtype=np.float64).ravel()

    if len(y) != len(z):
        raise ValueError(f"Shape mismatch: y_true ({y.shape}) vs y_pred_margin ({z.shape})")

    p = sigmoid(z)
    factor = 1.0 + (w - 1.0) * y

    # First derivative (gradient)
    grad = p * factor - w * y

    # Second derivative (Hessian) with damping epsilon
    hess = p * (1.0 - p) * factor + epsilon

    # Guarantee strict positivity of Hessian
    hess = np.maximum(hess, epsilon)

    return grad, hess


def cost_sensitive_gradient(
    y_true: Union[float, np.ndarray],
    y_pred_margin: Union[float, np.ndarray],
    w: float = 5.0,
) -> np.ndarray:
    """
    Compute analytical first-order gradient g_i = dL/dy_hat.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_pred_margin : array-like
        Predicted raw margins (logits).
    w : float, default=5.0
        Cost penalty multiplier.

    Returns
    -------
    np.ndarray
        First-order gradients g_i.
    """
    grad, _ = cost_sensitive_gradients(y_true, y_pred_margin, w=w, epsilon=0.0)
    return grad


def cost_sensitive_hessian(
    y_true: Union[float, np.ndarray],
    y_pred_margin: Union[float, np.ndarray],
    w: float = 5.0,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """
    Compute analytical damped second-order Hessian h_i = d^2L/dy_hat^2 + epsilon.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_pred_margin : array-like
        Predicted raw margins (logits).
    w : float, default=5.0
        Cost penalty multiplier.
    epsilon : float, default=1e-6
        Hessian damping factor.

    Returns
    -------
    np.ndarray
        Damped second-order Hessians h_i.
    """
    _, hess = cost_sensitive_gradients(y_true, y_pred_margin, w=w, epsilon=epsilon)
    return hess


def numerical_gradient(
    y_true: np.ndarray,
    y_pred_margin: np.ndarray,
    w: float = 5.0,
    delta: float = 1e-6,
) -> np.ndarray:
    """
    Compute numerical finite-difference approximation of the first-order gradient.

    Uses central difference:
        g_num = [ L(y, z + delta) - L(y, z - delta) ] / (2 * delta)

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_pred_margin : np.ndarray
        Predicted raw margins.
    w : float, default=5.0
        Cost penalty multiplier.
    delta : float, default=1e-6
        Step size for central difference.

    Returns
    -------
    np.ndarray
        Numerical gradient approximation.
    """
    y = np.asarray(y_true, dtype=np.float64).ravel()
    z = np.asarray(y_pred_margin, dtype=np.float64).ravel()

    loss_plus = cost_sensitive_loss_elementwise(y, z + delta, w=w)
    loss_minus = cost_sensitive_loss_elementwise(y, z - delta, w=w)

    return (loss_plus - loss_minus) / (2.0 * delta)


def numerical_hessian(
    y_true: np.ndarray,
    y_pred_margin: np.ndarray,
    w: float = 5.0,
    delta: float = 1e-6,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """
    Compute numerical finite-difference approximation of the second-order Hessian.

    Uses central difference on analytical gradient g(z):
        h_num = [ g(z + delta) - g(z - delta) ] / (2 * delta) + epsilon

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_pred_margin : np.ndarray
        Predicted raw margins.
    w : float, default=5.0
        Cost penalty multiplier.
    delta : float, default=1e-6
        Step size for central difference.
    epsilon : float, default=1e-6
        Hessian damping factor.

    Returns
    -------
    np.ndarray
        Numerical Hessian approximation.
    """
    y = np.asarray(y_true, dtype=np.float64).ravel()
    z = np.asarray(y_pred_margin, dtype=np.float64).ravel()

    grad_plus = cost_sensitive_gradient(y, z + delta, w=w)
    grad_minus = cost_sensitive_gradient(y, z - delta, w=w)

    return (grad_plus - grad_minus) / (2.0 * delta) + epsilon
