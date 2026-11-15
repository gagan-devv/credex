"""
Replication Suite for 9 Baseline Machine Learning Algorithms.

Implements unified scikit-learn compatible estimator wrappers for all 9 algorithms
evaluated in Li & Wu (2024), "Loan default predictability with explainable machine learning",
Finance Research Letters, 60, 104867:
    1. Logistic Regression (L2 penalty)
    2. Decision Tree Classifier (CART)
    3. Random Forest Classifier (Paper champion: ROC-AUC ~ 0.835)
    4. AdaBoost Classifier
    5. Gradient Boosting Decision Tree (GBDT / Scikit-Learn)
    6. XGBoost Classifier
    7. LightGBM Classifier (Paper runner-up: ROC-AUC ~ 0.831)
    8. CatBoost Classifier
    9. Multilayer Perceptron (MLP Neural Network)
"""

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)

# Optional external gradient boosting libraries with graceful fallback
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None


# Canonical model names as defined in configs/models_config.yaml
BASELINE_MODEL_NAMES: Tuple[str, ...] = (
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "adaboost",
    "gbdt",
    "xgboost",
    "lightgbm",
    "catboost",
    "mlp",
)

# Published benchmark targets from Li & Wu (2024) Table 2
PUBLISHED_BENCHMARKS: Dict[str, Dict[str, float]] = {
    "random_forest": {"roc_auc": 0.835},
    "lightgbm": {"roc_auc": 0.831},
}

# Standard default hyperparameters from configs/models_config.yaml
DEFAULT_BASELINE_PARAMS: Dict[str, Dict[str, Any]] = {
    "logistic_regression": {
        "penalty": "l2",
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": 42,
    },
    "decision_tree": {
        "criterion": "gini",
        "max_depth": 10,
        "min_samples_split": 50,
        "min_samples_leaf": 20,
        "random_state": 42,
    },
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "n_jobs": -1,
        "random_state": 42,
    },
    "adaboost": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "random_state": 42,
    },
    "gbdt": {
        "n_estimators": 150,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "random_state": 42,
    },
    "xgboost": {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "random_state": 42,
    },
    "lightgbm": {
        "n_estimators": 250,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1,
    },
    "catboost": {
        "iterations": 300,
        "learning_rate": 0.05,
        "depth": 6,
        "random_seed": 42,
        "verbose": 0,
    },
    "mlp": {
        "hidden_layer_sizes": (64, 32),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.0001,
        "batch_size": 256,
        "learning_rate_init": 0.001,
        "max_iter": 200,
        "random_state": 42,
    },
}


def _create_raw_estimator(model_name: str, params: Dict[str, Any]) -> BaseEstimator:
    """Instantiate underlying estimator object for the given model_name."""
    name_clean = model_name.lower().strip()
    p = copy.deepcopy(params)

    if name_clean == "logistic_regression":
        return LogisticRegression(**p)

    elif name_clean == "decision_tree":
        return DecisionTreeClassifier(**p)

    elif name_clean == "random_forest":
        return RandomForestClassifier(**p)

    elif name_clean == "adaboost":
        return AdaBoostClassifier(**p)

    elif name_clean == "gbdt":
        return GradientBoostingClassifier(**p)

    elif name_clean == "xgboost":
        if XGBClassifier is None:
            raise ImportError(
                "xgboost is not installed or failed to import. "
                "Please ensure xgboost>=2.0.0 is available in your environment."
            )
        return XGBClassifier(**p)

    elif name_clean == "lightgbm":
        if LGBMClassifier is None:
            raise ImportError(
                "lightgbm is not installed or failed to import. "
                "Please ensure lightgbm>=4.1.0 is available in your environment."
            )
        return LGBMClassifier(**p)

    elif name_clean == "catboost":
        if CatBoostClassifier is None:
            raise ImportError(
                "catboost is not installed or failed to import. "
                "Please ensure catboost>=1.2.0 is available in your environment."
            )
        if "random_state" in p and "random_seed" in p:
            del p["random_seed"]
        return CatBoostClassifier(**p)

    elif name_clean == "mlp":
        if "hidden_layer_sizes" in p and isinstance(p["hidden_layer_sizes"], list):
            p["hidden_layer_sizes"] = tuple(p["hidden_layer_sizes"])
        return MLPClassifier(**p)

    else:
        valid_models = ", ".join(BASELINE_MODEL_NAMES)
        raise ValueError(
            f"Unsupported model name '{model_name}'. Must be one of: {valid_models}"
        )


class BaselineModel(BaseEstimator, ClassifierMixin):
    """
    Unified scikit-learn compatible estimator wrapper for the 9 baseline
    credit default prediction algorithms evaluated in Li & Wu (2024).

    Parameters
    ----------
    model_name : str
        One of the 9 canonical baseline model identifiers:
        'logistic_regression', 'decision_tree', 'random_forest', 'adaboost',
        'gbdt', 'xgboost', 'lightgbm', 'catboost', 'mlp'.
    **params : Any
        Hyperparameters passed directly to the underlying model constructor.
        Any omitted parameters fall back to defaults specified in models_config.yaml.
    """

    def __init__(self, model_name: str = "logistic_regression", **params: Any) -> None:
        self.model_name = model_name.lower().strip()
        self.params = params

        # Validate model name
        if self.model_name not in BASELINE_MODEL_NAMES:
            valid_models = ", ".join(BASELINE_MODEL_NAMES)
            raise ValueError(
                f"Unknown model_name '{self.model_name}'. Must be one of: {valid_models}"
            )

        # Merge defaults with custom parameters
        merged_params = copy.deepcopy(DEFAULT_BASELINE_PARAMS.get(self.model_name, {}))
        merged_params.update(self.params)
        self._effective_params = merged_params

        # Build raw estimator
        self.estimator = _create_raw_estimator(self.model_name, self._effective_params)
        self.classes_ = np.array([0, 1])
        self.is_fitted_ = False
        self.feature_names_in_: Optional[List[str]] = None
        self.n_features_in_: Optional[int] = None

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator (scikit-learn convention)."""
        res = {"model_name": self.model_name}
        res.update(self.params)
        return res

    def __sklearn_clone__(self) -> "BaselineModel":
        """Support scikit-learn clone() without requiring rigid constructor args."""
        return BaselineModel(model_name=self.model_name, **copy.deepcopy(self.params))

    def set_params(self, **params: Any) -> "BaselineModel":
        """Set the parameters of this estimator."""
        if "model_name" in params:
            self.model_name = params.pop("model_name").lower().strip()
        self.params.update(params)

        merged_params = copy.deepcopy(DEFAULT_BASELINE_PARAMS.get(self.model_name, {}))
        merged_params.update(self.params)
        self._effective_params = merged_params

        self.estimator = _create_raw_estimator(self.model_name, self._effective_params)
        self.is_fitted_ = False
        return self

    @property
    def underlying_estimator(self) -> BaseEstimator:
        """Access the underlying native estimator object."""
        return self.estimator

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        **fit_params: Any,
    ) -> "BaselineModel":
        """
        Fit the baseline model on training features X and binary target y.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y : pd.Series or np.ndarray of shape (n_samples,)
            Binary target labels {0, 1}.
        **fit_params : Any
            Additional keyword arguments passed to the underlying fit method.

        Returns
        -------
        self : BaselineModel
            Fitted estimator instance.
        """
        # Validate and record feature metadata
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
            self.n_features_in_ = X.shape[1]
            X_clean = X.copy()
        elif isinstance(X, np.ndarray):
            self.feature_names_in_ = None
            self.n_features_in_ = X.shape[1] if X.ndim > 1 else 1
            X_clean = X.copy()
        else:
            X_arr = np.asarray(X)
            self.feature_names_in_ = None
            self.n_features_in_ = X_arr.shape[1] if X_arr.ndim > 1 else 1
            X_clean = X_arr

        # Format target labels
        if isinstance(y, pd.Series):
            y_clean = y.to_numpy(dtype=int)
        else:
            y_clean = np.asarray(y, dtype=int)

        # Validate binary classes
        unique_classes = np.unique(y_clean)
        if not np.all(np.isin(unique_classes, [0, 1])):
            raise ValueError(
                f"Targets must be binary {0, 1}. Found unique classes: {unique_classes}"
            )

        self.classes_ = np.array([0, 1])

        # Fit underlying model
        self.estimator.fit(X_clean, y_clean, **fit_params)
        self.is_fitted_ = True

        # Copy over fitted metadata if available
        if hasattr(self.estimator, "classes_"):
            self.classes_ = np.asarray(self.estimator.classes_)
        if hasattr(self.estimator, "n_features_in_"):
            self.n_features_in_ = self.estimator.n_features_in_

        return self

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Compute class probabilities for feature matrix X.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray of shape (n_samples, n_features)
            Predictor variables.

        Returns
        -------
        probs : np.ndarray of shape (n_samples, 2)
            Calibrated class probabilities [P(Y=0), P(Y=1)].
        """
        if not self.is_fitted_:
            raise ValueError(
                f"This {self.__class__.__name__} ('{self.model_name}') is not fitted yet. "
                "Call 'fit' before using this estimator."
            )

        if hasattr(self.estimator, "predict_proba"):
            probs = self.estimator.predict_proba(X)
        elif hasattr(self.estimator, "decision_function"):
            # For estimators with only decision_function, apply standard sigmoid link
            scores = self.estimator.decision_function(X)
            if scores.ndim == 1:
                p1 = 1.0 / (1.0 + np.exp(-scores))
                p0 = 1.0 - p1
                probs = np.column_stack([p0, p1])
            else:
                exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
                probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        else:
            # Fallback to binary predictions
            preds = self.estimator.predict(X)
            probs = np.column_stack([1.0 - preds, preds])

        probs = np.asarray(probs, dtype=float)

        # Handle binary 1D output edge case
        if probs.ndim == 1:
            p1 = np.clip(probs, 0.0, 1.0)
            p0 = 1.0 - p1
            probs = np.column_stack([p0, p1])
        elif probs.shape[1] == 1:
            p1 = np.clip(probs[:, 0], 0.0, 1.0)
            p0 = 1.0 - p1
            probs = np.column_stack([p0, p1])

        # Enforce strict probability properties: [0, 1] bounds and row normalization
        probs = np.clip(probs, 0.0, 1.0)
        row_sums = probs.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        probs = probs / row_sums

        return probs

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Predict binary default classification {0, 1} for feature matrix X.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray of shape (n_samples, n_features)
            Predictor variables.
        threshold : float, default=0.5
            Probability cutoff threshold tau in (0, 1). If P(Y=1) >= threshold,
            predict 1 (default), else 0 (repay).

        Returns
        -------
        preds : np.ndarray of shape (n_samples,)
            Binary predictions in {0, 1}.
        """
        if not (0.0 < threshold < 1.0):
            raise ValueError(
                f"Classification threshold must be strictly within (0, 1). Got: {threshold}"
            )

        probs = self.predict_proba(X)
        p1 = probs[:, 1]
        return (p1 >= threshold).astype(int)


def get_baseline_model(
    model_name: str,
    config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> BaselineModel:
    """
    Factory function to instantiate a configured BaselineModel.

    Parameters
    ----------
    model_name : str
        One of the 9 algorithms in BASELINE_MODEL_NAMES.
    config : Optional[Dict[str, Any]], optional
        Full models config dictionary or models['baselines'] sub-dictionary.
    **kwargs : Any
        Explicit hyperparameter overrides.

    Returns
    -------
    model : BaselineModel
        Configured BaselineModel instance.
    """
    clean_name = model_name.lower().strip()
    if clean_name not in BASELINE_MODEL_NAMES:
        valid_models = ", ".join(BASELINE_MODEL_NAMES)
        raise ValueError(
            f"Unsupported model name '{model_name}'. Must be one of: {valid_models}"
        )

    # Extract base parameters from config if provided
    base_params: Dict[str, Any] = {}
    if config is not None:
        if "baselines" in config and clean_name in config["baselines"]:
            base_params = copy.deepcopy(config["baselines"][clean_name])
        elif clean_name in config:
            base_params = copy.deepcopy(config[clean_name])

    # Overlay explicit kwargs
    base_params.update(kwargs)

    return BaselineModel(model_name=clean_name, **base_params)


def get_all_baseline_models(
    config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, BaselineModel]:
    """
    Instantiate all 9 baseline algorithms under standard symmetric cross-entropy.

    Parameters
    ----------
    config : Optional[Dict[str, Any]], optional
        Optional models configuration dictionary.
    **kwargs : Any
        Common hyperparameter overrides (e.g., random_state=42).

    Returns
    -------
    models : Dict[str, BaselineModel]
        Dictionary mapping canonical model names to instantiated BaselineModel objects.
    """
    models: Dict[str, BaselineModel] = {}
    for name in BASELINE_MODEL_NAMES:
        models[name] = get_baseline_model(name, config=config, **kwargs)
    return models


def run_baseline_replication(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    cv: int = 5,
    random_state: int = 42,
    models: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute replication benchmark for baseline algorithms across Stratified K-Fold CV.

    Computes ROC-AUC, PR-AUC, Brier score, Accuracy, F1, Precision, and Recall under
    standard binary cross-entropy loss, evaluating replication against Table 2 of
    Li & Wu (2024).

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    y : pd.Series or np.ndarray
        Binary target labels {0, 1}.
    cv : int, default=5
        Number of stratified cross-validation folds.
    random_state : int, default=42
        Random seed for reproducibility.
    models : Optional[List[str]], optional
        Subset of baseline models to evaluate. Defaults to all 9 algorithms.
    config : Optional[Dict[str, Any]], optional
        Optional configuration dictionary.

    Returns
    -------
    results : Dict[str, Any]
        Replication metrics, benchmark comparisons, and per-fold performance.
    """
    target_models = models or list(BASELINE_MODEL_NAMES)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    X_mat = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
    y_vec = y.to_numpy(dtype=int) if isinstance(y, pd.Series) else np.asarray(y, dtype=int)

    benchmark_results: Dict[str, Any] = {}

    for model_name in target_models:
        logger.info("Evaluating baseline model: %s", model_name)
        fold_metrics: Dict[str, List[float]] = {
            "roc_auc": [],
            "pr_auc": [],
            "brier_score": [],
            "accuracy": [],
            "f1": [],
            "precision": [],
            "recall": [],
        }

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_mat, y_vec)):
            X_tr, y_tr = X_mat[train_idx], y_vec[train_idx]
            X_val, y_val = X_mat[val_idx], y_vec[val_idx]

            model = get_baseline_model(model_name, config=config, random_state=random_state)
            model.fit(X_tr, y_tr)

            probs = model.predict_proba(X_val)
            p1 = probs[:, 1]
            preds = model.predict(X_val, threshold=0.5)

            if len(np.unique(y_val)) > 1:
                fold_metrics["roc_auc"].append(float(roc_auc_score(y_val, p1)))
                fold_metrics["pr_auc"].append(float(average_precision_score(y_val, p1)))
            else:
                fold_metrics["roc_auc"].append(0.5)
                fold_metrics["pr_auc"].append(0.0)
            fold_metrics["brier_score"].append(float(brier_score_loss(y_val, p1)))
            fold_metrics["accuracy"].append(float(accuracy_score(y_val, preds)))
            fold_metrics["f1"].append(float(f1_score(y_val, preds, zero_division=0)))
            fold_metrics["precision"].append(float(precision_score(y_val, preds, zero_division=0)))
            fold_metrics["recall"].append(float(recall_score(y_val, preds, zero_division=0)))

        summary: Dict[str, Dict[str, float]] = {}
        for metric, values in fold_metrics.items():
            summary[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

        # Compare with paper published metrics if available
        comparison: Dict[str, Any] = {}
        if model_name in PUBLISHED_BENCHMARKS:
            target_auc = PUBLISHED_BENCHMARKS[model_name]["roc_auc"]
            achieved_auc = summary["roc_auc"]["mean"]
            delta = achieved_auc - target_auc
            comparison = {
                "target_roc_auc": target_auc,
                "achieved_roc_auc": achieved_auc,
                "delta": delta,
                "percentage_diff": (delta / target_auc) * 100.0,
            }

        benchmark_results[model_name] = {
            "summary": summary,
            "folds": fold_metrics,
            "published_benchmark_comparison": comparison,
        }

    return benchmark_results
