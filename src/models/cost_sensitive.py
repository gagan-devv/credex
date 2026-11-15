"""
Cost-Sensitive Gradient Boosted Ensembles (LightGBM, XGBoost, CatBoost).

This module provides custom objective wrappers and scikit-learn compatible estimator classes
that inject asymmetric cost-sensitive loss gradients (w = C_FN / C_FP >= 5) directly into:
1. LightGBM (via custom `objective` / `fobj` callback)
2. XGBoost (via custom `objective` callback)
3. CatBoost (via custom `loss_function` object implementing `calc_ders_range`)

These objectives heavily penalize False Negatives (defaulting loans classified as good)
at tree-split time, resulting in higher default recall and lower economic loss.
"""

from typing import Any, Callable, List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from src.models.objective_functions import (
    cost_sensitive_gradients,
    cost_sensitive_loss,
    sigmoid,
)

# Optional imports for tree libraries
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import catboost as cb
except ImportError:
    cb = None


# ==============================================================================
# LightGBM Objective & Metric Callbacks
# ==============================================================================


def make_lightgbm_objective(
    w: float = 5.0,
    epsilon: float = 1e-6,
) -> Callable[..., Tuple[np.ndarray, np.ndarray]]:
    """
    Create a custom objective callback for LightGBM.

    Handles both:
    1. Scikit-learn API (LGBMClassifier): objective(y_true, y_pred)
    2. Low-level API (lightgbm.train): fobj(preds, dataset)

    Parameters
    ----------
    w : float, default=5.0
        Cost penalty ratio C_FN / C_FP.
    epsilon : float, default=1e-6
        Hessian damping factor.

    Returns
    -------
    Callable
        Custom objective function for LightGBM.
    """

    def lightgbm_objective(arg1: Any, arg2: Any) -> Tuple[np.ndarray, np.ndarray]:
        if hasattr(arg2, "get_label"):
            # Called as (preds, dataset)
            preds = arg1
            labels = arg2.get_label()
        elif hasattr(arg1, "get_label"):
            # Called as (dataset, preds)
            labels = arg1.get_label()
            preds = arg2
        else:
            # Called as (y_true, y_pred) by LGBMClassifier
            labels = arg1
            preds = arg2

        grad, hess = cost_sensitive_gradients(labels, preds, w=w, epsilon=epsilon)
        return grad, hess

    return lightgbm_objective


def make_lightgbm_eval_metric(
    w: float = 5.0,
) -> Callable[..., Tuple[str, float, bool]]:
    """
    Create a custom evaluation metric callback for LightGBM.

    Parameters
    ----------
    w : float, default=5.0
        Cost penalty ratio C_FN / C_FP.

    Returns
    -------
    Callable
        Custom evaluation metric returning ('cost_loss', eval_result, False).
    """

    def lightgbm_eval_metric(arg1: Any, arg2: Any) -> Tuple[str, float, bool]:
        if hasattr(arg2, "get_label"):
            preds = arg1
            labels = arg2.get_label()
        elif hasattr(arg1, "get_label"):
            labels = arg1.get_label()
            preds = arg2
        else:
            labels = arg1
            preds = arg2

        probs = sigmoid(preds)
        loss = cost_sensitive_loss(labels, probs, w=w)
        return "cost_sensitive_loss", loss, False

    return lightgbm_eval_metric


# ==============================================================================
# XGBoost Objective & Metric Callbacks
# ==============================================================================


def make_xgboost_objective(
    w: float = 5.0,
    epsilon: float = 1e-6,
) -> Callable[..., Tuple[np.ndarray, np.ndarray]]:
    """
    Create a custom objective callback for XGBoost.

    Handles both:
    1. Scikit-learn API (XGBClassifier): objective(y_true, y_pred)
    2. Low-level API (xgb.train): objective(preds, dtrain)

    Parameters
    ----------
    w : float, default=5.0
        Cost penalty ratio C_FN / C_FP.
    epsilon : float, default=1e-6
        Hessian damping factor.

    Returns
    -------
    Callable
        Custom objective function for XGBoost.
    """

    def xgboost_objective(arg1: Any, arg2: Any) -> Tuple[np.ndarray, np.ndarray]:
        if hasattr(arg2, "get_label"):
            # Called as (preds, dtrain)
            preds = arg1
            labels = arg2.get_label()
        elif hasattr(arg1, "get_label"):
            labels = arg1.get_label()
            preds = arg2
        else:
            # Called as (y_true, y_pred) by XGBClassifier
            labels = arg1
            preds = arg2

        grad, hess = cost_sensitive_gradients(labels, preds, w=w, epsilon=epsilon)
        return grad, hess

    return xgboost_objective


def make_xgboost_eval_metric(
    w: float = 5.0,
) -> Callable[..., Tuple[str, float]]:
    """
    Create a custom evaluation metric callback for XGBoost.

    Parameters
    ----------
    w : float, default=5.0
        Cost penalty ratio C_FN / C_FP.

    Returns
    -------
    Callable
        Custom evaluation metric returning ('cost_loss', eval_result).
    """

    def xgboost_eval_metric(arg1: Any, arg2: Any) -> Tuple[str, float]:
        if hasattr(arg2, "get_label"):
            preds = arg1
            labels = arg2.get_label()
        elif hasattr(arg1, "get_label"):
            labels = arg1.get_label()
            preds = arg2
        else:
            labels = arg1
            preds = arg2

        probs = sigmoid(preds)
        loss = cost_sensitive_loss(labels, probs, w=w)
        return "cost_sensitive_loss", loss

    return xgboost_eval_metric


# ==============================================================================
# CatBoost Custom Objective Class
# ==============================================================================


class CostSensitiveCatBoostObjective:
    """
    Custom objective object for CatBoost implementing `calc_ders_range`.

    In CatBoost:
        der1 = - dL/dy_hat  (negative gradient)
        der2 = - d^2L/dy_hat^2  (negative Hessian)
    """

    def __init__(self, w: float = 5.0, epsilon: float = 1e-6):
        self.w = float(w)
        self.epsilon = float(epsilon)

    def calc_ders_range(
        self,
        approxes: Any,
        targets: Any,
        weights: Optional[Any],
    ) -> List[Tuple[float, float]]:
        """
        Calculate first and second order derivatives for CatBoost.

        Parameters
        ----------
        approxes : array-like
            Raw margin logits from the current ensemble.
        targets : array-like
            Binary ground truth labels (0 or 1).
        weights : array-like, optional
            Instance sample weights.

        Returns
        -------
        List[Tuple[float, float]]
            List of (der1, der2) pairs for each sample.
        """
        approx_arr = np.asarray(approxes, dtype=np.float64)
        target_arr = np.asarray(targets, dtype=np.float64)

        grad, hess = cost_sensitive_gradients(
            target_arr,
            approx_arr,
            w=self.w,
            epsilon=self.epsilon,
        )

        # CatBoost expects negative gradient and negative curvature
        der1 = -grad
        der2 = -hess

        if weights is not None:
            w_arr = np.asarray(weights, dtype=np.float64)
            der1 = der1 * w_arr
            der2 = der2 * w_arr

        return list(zip(der1.tolist(), der2.tolist()))


# ==============================================================================
# Scikit-Learn Compatible Estimator Wrappers
# ==============================================================================


class CostSensitiveLGBMClassifier(BaseEstimator, ClassifierMixin):
    """
    Cost-Sensitive LightGBM Classifier with custom asymmetric gradient loss.

    Parameters
    ----------
    w : float, default=5.0
        Cost penalty ratio C_FN / C_FP.
    epsilon : float, default=1e-6
        Hessian damping factor.
    n_estimators : int, default=100
        Number of boosting trees.
    learning_rate : float, default=0.05
        Boosting learning rate.
    max_depth : int, default=-1
        Maximum tree depth.
    num_leaves : int, default=31
        Maximum tree leaves.
    random_state : int, default=42
        Random seed for reproducibility.
    **kwargs : dict
        Additional parameters passed to LightGBM.
    """

    def __init__(
        self,
        w: float = 5.0,
        epsilon: float = 1e-6,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = -1,
        num_leaves: int = 31,
        random_state: int = 42,
        **kwargs: Any,
    ):
        self.w = w
        self.epsilon = epsilon
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.random_state = random_state
        self.kwargs = kwargs
        self.model_ = None
        self.classes_ = np.array([0, 1])

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Optional[np.ndarray] = None,
        **fit_params: Any,
    ) -> "CostSensitiveLGBMClassifier":
        if lgb is None:
            raise ImportError("lightgbm is required for CostSensitiveLGBMClassifier.")

        custom_obj = make_lightgbm_objective(w=self.w, epsilon=self.epsilon)

        params = {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "num_leaves": self.num_leaves,
            "random_state": self.random_state,
            "objective": custom_obj,
            "verbose": -1,
            **self.kwargs,
        }

        self.model_ = lgb.LGBMClassifier(**params)
        self.model_.fit(X, y, sample_weight=sample_weight, **fit_params)
        return self

    def decision_function(self, X: Any) -> np.ndarray:
        """Return raw margin predictions (logits)."""
        if self.model_ is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.model_.predict(X, raw_score=True)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities via calibrated sigmoid of margins."""
        margins = self.decision_function(X)
        probs_pos = sigmoid(margins)
        probs_neg = 1.0 - probs_pos
        return np.column_stack([probs_neg, probs_pos])

    def predict(self, X: Any, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class labels based on threshold."""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)


class CostSensitiveXGBClassifier(BaseEstimator, ClassifierMixin):
    """
    Cost-Sensitive XGBoost Classifier with custom asymmetric gradient loss.

    Parameters
    ----------
    w : float, default=5.0
        Cost penalty ratio C_FN / C_FP.
    epsilon : float, default=1e-6
        Hessian damping factor.
    n_estimators : int, default=100
        Number of boosting trees.
    learning_rate : float, default=0.05
        Boosting learning rate.
    max_depth : int, default=6
        Maximum tree depth.
    random_state : int, default=42
        Random seed for reproducibility.
    **kwargs : dict
        Additional parameters passed to XGBoost.
    """

    def __init__(
        self,
        w: float = 5.0,
        epsilon: float = 1e-6,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        random_state: int = 42,
        **kwargs: Any,
    ):
        self.w = w
        self.epsilon = epsilon
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.kwargs = kwargs
        self.model_ = None
        self.classes_ = np.array([0, 1])

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Optional[np.ndarray] = None,
        **fit_params: Any,
    ) -> "CostSensitiveXGBClassifier":
        if xgb is None:
            raise ImportError("xgboost is required for CostSensitiveXGBClassifier.")

        custom_obj = make_xgboost_objective(w=self.w, epsilon=self.epsilon)

        params = {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "random_state": self.random_state,
            "objective": custom_obj,
            **self.kwargs,
        }

        self.model_ = xgb.XGBClassifier(**params)
        self.model_.fit(X, y, sample_weight=sample_weight, **fit_params)
        return self

    def decision_function(self, X: Any) -> np.ndarray:
        """Return raw margin predictions (logits)."""
        if self.model_ is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.model_.predict(X, output_margin=True)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities via calibrated sigmoid of margins."""
        margins = self.decision_function(X)
        probs_pos = sigmoid(margins)
        probs_neg = 1.0 - probs_pos
        return np.column_stack([probs_neg, probs_pos])

    def predict(self, X: Any, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class labels based on threshold."""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)


class CostSensitiveCatBoostClassifier(BaseEstimator, ClassifierMixin):
    """
    Cost-Sensitive CatBoost Classifier with custom asymmetric gradient loss.

    Parameters
    ----------
    w : float, default=5.0
        Cost penalty ratio C_FN / C_FP.
    epsilon : float, default=1e-6
        Hessian damping factor.
    iterations : int, default=100
        Number of boosting trees.
    learning_rate : float, default=0.05
        Boosting learning rate.
    depth : int, default=6
        Tree depth.
    random_seed : int, default=42
        Random seed for reproducibility.
    eval_metric : str, default="Logloss"
        Evaluation metric required by CatBoost for custom loss functions.
    **kwargs : dict
        Additional parameters passed to CatBoost.
    """

    def __init__(
        self,
        w: float = 5.0,
        epsilon: float = 1e-6,
        iterations: int = 100,
        learning_rate: float = 0.05,
        depth: int = 6,
        random_seed: int = 42,
        eval_metric: str = "Logloss",
        verbose: bool = False,
        **kwargs: Any,
    ):
        self.w = w
        self.epsilon = epsilon
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.random_seed = random_seed
        self.eval_metric = eval_metric
        self.verbose = verbose
        self.kwargs = kwargs
        self.model_ = None
        self.classes_ = np.array([0, 1])

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Optional[np.ndarray] = None,
        **fit_params: Any,
    ) -> "CostSensitiveCatBoostClassifier":
        if cb is None:
            raise ImportError("catboost is required for CostSensitiveCatBoostClassifier.")

        custom_obj = CostSensitiveCatBoostObjective(w=self.w, epsilon=self.epsilon)

        params = {
            "iterations": self.iterations,
            "learning_rate": self.learning_rate,
            "depth": self.depth,
            "random_seed": self.random_seed,
            "loss_function": custom_obj,
            "eval_metric": self.eval_metric,
            "verbose": self.verbose,
            "train_dir": "/tmp/catboost_info",
            **self.kwargs,
        }

        self.model_ = cb.CatBoostClassifier(**params)
        self.model_.fit(X, y, sample_weight=sample_weight, **fit_params)
        return self

    def decision_function(self, X: Any) -> np.ndarray:
        """Return raw margin predictions (logits)."""
        if self.model_ is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.model_.predict(X, prediction_type="RawFormulaVal")

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities via calibrated sigmoid of margins."""
        margins = self.decision_function(X)
        probs_pos = sigmoid(margins)
        probs_neg = 1.0 - probs_pos
        return np.column_stack([probs_neg, probs_pos])

    def predict(self, X: Any, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class labels based on threshold."""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)
