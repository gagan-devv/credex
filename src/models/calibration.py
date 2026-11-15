"""
Probability Calibration Engine for Credit Default Models.

Implements probability calibration methods (Isotonic Regression and Platt Scaling)
and evaluation metrics (Brier Score, Expected Calibration Error, Calibration Curves).

In asymmetric cost-sensitive credit risk models, raw model margins and probability estimates
can deviate from true empirical posterior default probabilities P(Y=1|X).
Out-of-fold calibration maps model scores to well-calibrated empirical default frequencies,
guaranteeing calibrated probabilities remain strictly bounded within [0, 1].
"""

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


def compute_brier_score(
    y_true: Union[pd.Series, np.ndarray, List[int]],
    y_prob: Union[pd.Series, np.ndarray, List[float]],
) -> float:
    """
    Calculate the Brier score: mean squared difference between predicted probabilities
    and actual binary outcomes.

    Brier = (1 / N) * sum_{i=1}^N (p_i - y_i)^2

    Parameters
    ----------
    y_true : pd.Series or np.ndarray of shape (n_samples,)
        True binary labels {0, 1}.
    y_prob : pd.Series or np.ndarray of shape (n_samples,)
        Predicted probability of the positive class (1). Must be in [0, 1].

    Returns
    -------
    score : float
        Brier score in [0, 1]. Lower is better (0 is perfect calibration and accuracy).
    """
    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.asarray(y_prob, dtype=float).ravel()

    if y_t.shape[0] != y_p.shape[0]:
        raise ValueError(
            f"Shape mismatch: y_true has {y_t.shape[0]} samples, y_prob has {y_p.shape[0]}."
        )

    if len(y_t) == 0:
        raise ValueError("Cannot compute Brier score on empty inputs.")

    # Validate binary targets
    unique_vals = np.unique(y_t)
    if not np.all(np.isin(unique_vals, [0.0, 1.0])):
        raise ValueError(f"y_true must contain only binary values {{0, 1}}, got: {unique_vals}")

    # Clip probabilities strictly into [0, 1]
    y_p_clipped = np.clip(y_p, 0.0, 1.0)

    # Exact analytical calculation
    score = float(np.mean((y_p_clipped - y_t) ** 2))
    return score


def compute_expected_calibration_error(
    y_true: Union[pd.Series, np.ndarray],
    y_prob: Union[pd.Series, np.ndarray],
    n_bins: int = 10,
) -> float:
    """
    Calculate Expected Calibration Error (ECE) across partitioned probability bins.

    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|

    Parameters
    ----------
    y_true : pd.Series or np.ndarray of shape (n_samples,)
        True binary labels {0, 1}.
    y_prob : pd.Series or np.ndarray of shape (n_samples,)
        Predicted probability of the positive class.
    n_bins : int, default=10
        Number of equal-width bins spanning [0, 1].

    Returns
    -------
    ece : float
        Expected calibration error in [0, 1].
    """
    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.clip(np.asarray(y_prob, dtype=float).ravel(), 0.0, 1.0)
    n_samples = len(y_t)

    if n_samples == 0:
        return 0.0

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_p, bins) - 1
    # Handle upper boundary edge case: p = 1.0 goes into the last bin
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    for b in range(n_bins):
        mask = bin_indices == b
        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_acc = np.mean(y_t[mask])
            bin_conf = np.mean(y_p[mask])
            ece += (bin_count / n_samples) * np.abs(bin_acc - bin_conf)

    return float(ece)


def compute_calibration_curve(
    y_true: Union[pd.Series, np.ndarray],
    y_prob: Union[pd.Series, np.ndarray],
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute calibration curve: empirical default frequencies vs mean predicted probabilities.

    Parameters
    ----------
    y_true : pd.Series or np.ndarray
        True binary outcomes {0, 1}.
    y_prob : pd.Series or np.ndarray
        Predicted probabilities of class 1.
    n_bins : int, default=10
        Number of discretization bins.

    Returns
    -------
    prob_true : np.ndarray
        Empirical fraction of defaults in each non-empty bin.
    prob_pred : np.ndarray
        Mean predicted probability in each non-empty bin.
    """
    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.clip(np.asarray(y_prob, dtype=float).ravel(), 0.0, 1.0)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_p, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    prob_true_list: List[float] = []
    prob_pred_list: List[float] = []

    for b in range(n_bins):
        mask = bin_indices == b
        if np.sum(mask) > 0:
            prob_true_list.append(float(np.mean(y_t[mask])))
            prob_pred_list.append(float(np.mean(y_p[mask])))

    return np.array(prob_true_list), np.array(prob_pred_list)


class ProbabilityCalibrator:
    """
    Calibration engine mapping raw model predictions to well-calibrated probabilities.

    Supports:
      - 'isotonic': Non-parametric isotonic regression (monotonic piecewise constant).
      - 'platt' or 'sigmoid': Logistic calibration mapping log-odds via sigmoid link.

    All output probabilities are guaranteed to stay strictly within [0.0, 1.0].

    Parameters
    ----------
    method : str, default='isotonic'
        Calibration method: 'isotonic', 'platt', or 'sigmoid'.
    """

    def __init__(self, method: str = "isotonic") -> None:
        clean_method = method.lower().strip()
        if clean_method not in ["isotonic", "platt", "sigmoid"]:
            raise ValueError(
                f"Unknown calibration method '{method}'. Must be 'isotonic' or 'platt' ('sigmoid')."
            )
        self.method = "platt" if clean_method == "sigmoid" else clean_method
        self.is_fitted_ = False
        self._regressor: Optional[Union[IsotonicRegression, LogisticRegression]] = None

    def fit(
        self,
        y_prob: Union[pd.Series, np.ndarray],
        y_true: Union[pd.Series, np.ndarray],
    ) -> "ProbabilityCalibrator":
        """
        Fit the calibrator on raw probabilities and ground truth binary targets.

        Parameters
        ----------
        y_prob : pd.Series or np.ndarray of shape (n_samples,) or (n_samples, 2)
            Raw predicted probabilities of default.
        y_true : pd.Series or np.ndarray of shape (n_samples,)
            True binary targets {0, 1}.

        Returns
        -------
        self : ProbabilityCalibrator
            Fitted calibrator instance.
        """
        y_p = np.asarray(y_prob, dtype=float)
        # Extract class 1 if 2D array passed
        if y_p.ndim == 2:
            y_p = y_p[:, 1]
        y_p = np.clip(y_p.ravel(), 0.0, 1.0)

        y_t = np.asarray(y_true, dtype=int).ravel()

        if len(y_p) != len(y_t):
            raise ValueError(
                f"Length mismatch: y_prob has {len(y_p)}, y_true has {len(y_t)}."
            )

        if self.method == "isotonic":
            self._regressor = IsotonicRegression(
                y_min=0.0,
                y_max=1.0,
                out_of_bounds="clip",
                increasing=True,
            )
            self._regressor.fit(y_p, y_t)

        elif self.method == "platt":
            # Platt scaling: fit logistic regression on logit(p) with numerical stabilization
            eps = 1e-12
            p_clamped = np.clip(y_p, eps, 1.0 - eps)
            logits = np.log(p_clamped / (1.0 - p_clamped)).reshape(-1, 1)

            lr = LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000)
            lr.fit(logits, y_t)
            self._regressor = lr

        self.is_fitted_ = True
        return self

    def calibrate(
        self,
        y_prob: Union[pd.Series, np.ndarray],
    ) -> np.ndarray:
        """
        Calibrate probabilities, strictly bounded in [0.0, 1.0].

        Parameters
        ----------
        y_prob : pd.Series or np.ndarray
            Raw predicted probabilities of class 1.

        Returns
        -------
        calibrated_probs : np.ndarray of shape (n_samples,)
            Calibrated probabilities of class 1 in [0, 1].
        """
        if not self.is_fitted_ or self._regressor is None:
            raise ValueError("Calibrator is not fitted yet. Call 'fit' first.")

        y_p = np.asarray(y_prob, dtype=float)
        if y_p.ndim == 2:
            y_p = y_p[:, 1]
        y_p = np.clip(y_p.ravel(), 0.0, 1.0)

        if self.method == "isotonic":
            cal_probs = self._regressor.predict(y_p)
        else:
            # Platt scaling
            eps = 1e-12
            p_clamped = np.clip(y_p, eps, 1.0 - eps)
            logits = np.log(p_clamped / (1.0 - p_clamped)).reshape(-1, 1)
            cal_probs = self._regressor.predict_proba(logits)[:, 1]

        # Strictly enforce bounds and finite values
        cal_probs = np.nan_to_num(cal_probs, nan=0.5, posinf=1.0, neginf=0.0)
        return np.clip(cal_probs, 0.0, 1.0)

    def predict_proba(
        self,
        y_prob: Union[pd.Series, np.ndarray],
    ) -> np.ndarray:
        """
        Compute calibrated 2D class probabilities [P(Y=0), P(Y=1)].

        Parameters
        ----------
        y_prob : pd.Series or np.ndarray
            Raw predicted probabilities.

        Returns
        -------
        probs : np.ndarray of shape (n_samples, 2)
            Calibrated probabilities summing to 1.0.
        """
        p1 = self.calibrate(y_prob)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(
        self,
        y_prob: Union[pd.Series, np.ndarray],
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Predict binary labels based on calibrated probability and threshold tau.

        Parameters
        ----------
        y_prob : pd.Series or np.ndarray
            Raw probabilities.
        threshold : float, default=0.5
            Decision threshold in (0, 1).

        Returns
        -------
        preds : np.ndarray of shape (n_samples,)
            Binary predictions in {0, 1}.
        """
        if not (0.0 < threshold < 1.0):
            raise ValueError(
                f"Classification threshold must be in (0, 1). Got: {threshold}"
            )
        p1 = self.calibrate(y_prob)
        return (p1 >= threshold).astype(int)


class OutOfFoldCalibrator(BaseEstimator, ClassifierMixin):
    """
    Out-Of-Fold Probability Calibrator conforming to scikit-learn estimator interface.

    Fits an underlying base estimator across K stratified cross-validation folds,
    collects out-of-fold probability predictions, fits a ProbabilityCalibrator
    on those predictions, and finally trains the base estimator on the entire dataset.

    Parameters
    ----------
    base_estimator : BaseEstimator
        The base machine learning model to calibrate (e.g., BaselineModel, RandomForestClassifier).
    method : str, default='isotonic'
        Calibration algorithm ('isotonic' or 'platt').
    cv : int, default=5
        Number of cross-validation folds for out-of-fold calibration.
    random_state : int, default=42
        Seed for stratified K-fold partitioning.
    """

    def __init__(
        self,
        base_estimator: BaseEstimator,
        method: str = "isotonic",
        cv: int = 5,
        random_state: int = 42,
    ) -> None:
        self.base_estimator = base_estimator
        self.method = method
        self.cv = cv
        self.random_state = random_state

        self.calibrator = ProbabilityCalibrator(method=self.method)
        self.fitted_estimator_: Optional[BaseEstimator] = None
        self.classes_ = np.array([0, 1])
        self.is_fitted_ = False
        self.oof_predictions_: Optional[np.ndarray] = None

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        **fit_params: Any,
    ) -> "OutOfFoldCalibrator":
        """
        Fit the base estimator and calibrator using out-of-fold cross-validation.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y : pd.Series or np.ndarray of shape (n_samples,)
            Binary target labels {0, 1}.
        **fit_params : Any
            Additional keyword arguments passed to base_estimator.fit().

        Returns
        -------
        self : OutOfFoldCalibrator
            Fitted calibrator instance.
        """
        X_mat = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        y_vec = y.to_numpy(dtype=int) if isinstance(y, pd.Series) else np.asarray(y, dtype=int)

        n_samples = len(y_vec)
        oof_probs = np.zeros(n_samples, dtype=float)

        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        for train_idx, val_idx in skf.split(X_mat, y_vec):
            fold_estimator = clone(self.base_estimator)
            fold_estimator.fit(X_mat[train_idx], y_vec[train_idx], **fit_params)

            if hasattr(fold_estimator, "predict_proba"):
                probs = fold_estimator.predict_proba(X_mat[val_idx])
                p_val = probs[:, 1] if probs.ndim == 2 and probs.shape[1] > 1 else probs.ravel()
            elif hasattr(fold_estimator, "decision_function"):
                scores = fold_estimator.decision_function(X_mat[val_idx])
                p_val = 1.0 / (1.0 + np.exp(-scores))
            else:
                p_val = fold_estimator.predict(X_mat[val_idx]).astype(float)

            oof_probs[val_idx] = p_val

        self.oof_predictions_ = oof_probs

        # Fit probability calibrator on out-of-fold predictions
        self.calibrator = ProbabilityCalibrator(method=self.method)
        self.calibrator.fit(oof_probs, y_vec)

        # Train primary estimator on full dataset
        self.fitted_estimator_ = clone(self.base_estimator)
        self.fitted_estimator_.fit(X, y, **fit_params)

        self.classes_ = np.array([0, 1])
        if hasattr(self.fitted_estimator_, "classes_"):
            self.classes_ = np.asarray(self.fitted_estimator_.classes_)

        self.is_fitted_ = True
        return self

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Compute calibrated 2D class probabilities for feature matrix X.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Features to evaluate.

        Returns
        -------
        probs : np.ndarray of shape (n_samples, 2)
            Calibrated [P(Y=0), P(Y=1)], strictly in [0, 1] and summing to 1.
        """
        if not self.is_fitted_ or self.fitted_estimator_ is None:
            raise ValueError("OutOfFoldCalibrator is not fitted yet. Call 'fit' first.")

        if hasattr(self.fitted_estimator_, "predict_proba"):
            probs = self.fitted_estimator_.predict_proba(X)
            raw_p = probs[:, 1] if probs.ndim == 2 and probs.shape[1] > 1 else probs.ravel()
        elif hasattr(self.fitted_estimator_, "decision_function"):
            scores = self.fitted_estimator_.decision_function(X)
            raw_p = 1.0 / (1.0 + np.exp(-scores))
        else:
            raw_p = self.fitted_estimator_.predict(X).astype(float)

        return self.calibrator.predict_proba(raw_p)

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Predict binary classifications {0, 1} based on calibrated probabilities.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix.
        threshold : float, default=0.5
            Decision threshold in (0, 1).

        Returns
        -------
        preds : np.ndarray of shape (n_samples,)
            Binary predictions in {0, 1}.
        """
        probs = self.predict_proba(X)
        p1 = probs[:, 1]
        return (p1 >= threshold).astype(int)


def evaluate_calibration(
    y_true: Union[pd.Series, np.ndarray],
    y_prob_uncalibrated: Union[pd.Series, np.ndarray],
    y_prob_calibrated: Union[pd.Series, np.ndarray],
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of calibration improvement.

    Compares uncalibrated vs calibrated probabilities across Brier Score,
    Expected Calibration Error (ECE), Log-Loss, and calibration curve coordinates.

    Parameters
    ----------
    y_true : pd.Series or np.ndarray
        Ground truth binary outcomes {0, 1}.
    y_prob_uncalibrated : pd.Series or np.ndarray
        Raw, uncalibrated predicted probabilities of default.
    y_prob_calibrated : pd.Series or np.ndarray
        Calibrated probabilities of default.
    n_bins : int, default=10
        Number of discretization bins.

    Returns
    -------
    metrics : Dict[str, Any]
        Dictionary of comparative metrics and percentage improvements.
    """
    y_t = np.asarray(y_true, dtype=int).ravel()
    y_uncal = np.clip(np.asarray(y_prob_uncalibrated, dtype=float).ravel(), 0.0, 1.0)
    y_cal = np.clip(np.asarray(y_prob_calibrated, dtype=float).ravel(), 0.0, 1.0)

    # Brier Scores
    brier_uncal = compute_brier_score(y_t, y_uncal)
    brier_cal = compute_brier_score(y_t, y_cal)
    brier_improvement = ((brier_uncal - brier_cal) / max(brier_uncal, 1e-12)) * 100.0

    # Expected Calibration Errors
    ece_uncal = compute_expected_calibration_error(y_t, y_uncal, n_bins=n_bins)
    ece_cal = compute_expected_calibration_error(y_t, y_cal, n_bins=n_bins)
    ece_improvement = ((ece_uncal - ece_cal) / max(ece_uncal, 1e-12)) * 100.0

    # Log Losses
    eps = 1e-15
    y_uncal_safe = np.clip(y_uncal, eps, 1.0 - eps)
    y_cal_safe = np.clip(y_cal, eps, 1.0 - eps)
    ll_uncal = float(log_loss(y_t, y_uncal_safe))
    ll_cal = float(log_loss(y_t, y_cal_safe))

    # Curves
    curve_uncal_true, curve_uncal_pred = compute_calibration_curve(y_t, y_uncal, n_bins=n_bins)
    curve_cal_true, curve_cal_pred = compute_calibration_curve(y_t, y_cal, n_bins=n_bins)

    return {
        "brier_score": {
            "uncalibrated": brier_uncal,
            "calibrated": brier_cal,
            "improvement_pct": brier_improvement,
        },
        "expected_calibration_error": {
            "uncalibrated": ece_uncal,
            "calibrated": ece_cal,
            "improvement_pct": ece_improvement,
        },
        "log_loss": {
            "uncalibrated": ll_uncal,
            "calibrated": ll_cal,
        },
        "calibration_curves": {
            "uncalibrated": {
                "prob_true": curve_uncal_true.tolist(),
                "prob_pred": curve_uncal_pred.tolist(),
            },
            "calibrated": {
                "prob_true": curve_cal_true.tolist(),
                "prob_pred": curve_cal_pred.tolist(),
            },
        },
    }
