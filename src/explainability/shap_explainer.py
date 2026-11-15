"""
Fast TreeSHAP Attribution & Interaction Engine (CREDEX-16).

Computes exact local and global Shapley additive attributions using C++ optimized
TreeExplainer for tree-based ensemble models (LightGBM, XGBoost, CatBoost, Random Forest),
with strict additivity verification: phi_0 + sum(phi_j(x)) == f(x) within 1e-4 tolerance.
"""

from dataclasses import dataclass
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator

# Avoid matplotlib cache warning
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_credex")

logger = logging.getLogger(__name__)


def extract_tree_estimator(model: Any) -> Any:
    """
    Extract the underlying native tree estimator from wrapper classes.

    Supports:
    - BaselineModel: unwraps `underlying_estimator` or `estimator`
    - CostSensitiveLGBMClassifier / XGB / CatBoost: unwraps `model_`
    - Native scikit-learn, LightGBM, XGBoost, CatBoost estimators
    """
    if hasattr(model, "underlying_estimator") and model.underlying_estimator is not None:
        return model.underlying_estimator
    if hasattr(model, "model_") and model.model_ is not None:
        return model.model_
    if hasattr(model, "estimator") and model.estimator is not None:
        return model.estimator
    return model


def get_native_tree_output(estimator: Any, X: Any) -> np.ndarray:
    """
    Compute the native output f(x) of the tree estimator for additivity verification.

    For boosted trees (LGBM, XGBoost, CatBoost), TreeSHAP values sum to raw margin/logits.
    For Random Forest / Decision Trees, TreeSHAP values sum to predicted probabilities.
    """
    # 1. Random Forest / Extra Trees / Decision Tree (Probability space)
    est_type = estimator.__class__.__name__
    if any(k in est_type for k in ["RandomForest", "DecisionTree", "ExtraTrees"]):
        if hasattr(estimator, "predict_proba"):
            probs = estimator.predict_proba(X)
            if probs.ndim == 2 and probs.shape[1] >= 2:
                return probs[:, 1]
            return probs.ravel()

    # 2. LightGBM (Raw score / margin)
    if "LGBM" in est_type:
        if hasattr(estimator, "predict"):
            return estimator.predict(X, raw_score=True)

    # 3. XGBoost (Margin)
    if "XGB" in est_type:
        if hasattr(estimator, "predict"):
            try:
                return estimator.predict(X, output_margin=True)
            except TypeError:
                pass

    # 4. CatBoost (RawFormulaVal)
    if "CatBoost" in est_type:
        if hasattr(estimator, "predict"):
            return estimator.predict(X, prediction_type="RawFormulaVal")

    # 5. Generic decision_function
    if hasattr(estimator, "decision_function"):
        df_vals = estimator.decision_function(X)
        if df_vals.ndim == 2 and df_vals.shape[1] == 1:
            return df_vals.ravel()
        return df_vals

    # 6. Fallback to predict_proba or predict
    if hasattr(estimator, "predict_proba"):
        probs = estimator.predict_proba(X)
        if probs.ndim == 2 and probs.shape[1] >= 2:
            return probs[:, 1]
        return probs.ravel()

    return estimator.predict(X)


@dataclass
class TreeSHAPResult:
    """
    Structured container for TreeSHAP attribution outputs.

    Attributes:
    -----------
    explanation : shap.Explanation
        The standardized 2D Explanation object for the positive (default risk) class.
    shap_values : np.ndarray
        Shape (N, M) matrix of Shapley attribution values for class 1.
    base_values : np.ndarray
        Shape (N,) array of base values (expected model score phi_0).
    feature_names : List[str]
        List of M feature names corresponding to the columns of shap_values.
    data : np.ndarray
        Shape (N, M) input feature values evaluated.
    model_outputs : np.ndarray
        Shape (N,) native model outputs f(x).
    is_additive : bool
        Whether phi_0 + sum_j(phi_j(x)) == f(x) within numerical tolerance.
    max_additivity_error : float
        Maximum absolute discrepancy across all evaluated samples.
    global_importance : pd.Series
        Mean absolute Shapley values per feature, sorted descending.
    """

    explanation: shap.Explanation
    shap_values: np.ndarray
    base_values: np.ndarray
    feature_names: List[str]
    data: np.ndarray
    model_outputs: np.ndarray
    is_additive: bool
    max_additivity_error: float
    global_importance: pd.Series

    def get_top_features(self, n: int = 5) -> List[str]:
        """Return the names of the top N globally important features."""
        return self.global_importance.head(n).index.tolist()

    def to_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame of SHAP values with feature names as columns."""
        return pd.DataFrame(self.shap_values, columns=self.feature_names)

    def explain_instance(self, idx: int = 0) -> Dict[str, Any]:
        """
        Extract detailed local attribution for a specific evaluated instance.

        Returns dictionary containing:
        - index: integer sample index
        - base_value: expected model output phi_0
        - model_output: actual model prediction f(x)
        - feature_contributions: list of dicts {feature, value, shap_value} sorted by |shap|
        - top_adverse_factors: features pushing risk UP (shap_value > 0)
        - top_mitigating_factors: features pushing risk DOWN (shap_value < 0)
        """
        if idx < 0 or idx >= len(self.shap_values):
            raise IndexError(f"Instance index {idx} out of range [0, {len(self.shap_values) - 1}].")

        inst_shap = self.shap_values[idx]
        inst_data = self.data[idx]
        base_val = float(self.base_values[idx])
        out_val = float(self.model_outputs[idx])

        contributions = []
        for feat_name, val, sv in zip(self.feature_names, inst_data, inst_shap):
            contributions.append(
                {
                    "feature": feat_name,
                    "value": float(val) if np.issubdtype(type(val), np.number) else val,
                    "shap_value": float(sv),
                    "abs_shap": float(abs(sv)),
                }
            )

        # Sort by absolute SHAP attribution
        contributions_sorted = sorted(contributions, key=lambda c: c["abs_shap"], reverse=True)

        # Adverse factors (pushing risk towards default / positive SHAP)
        adverse = [c for c in contributions_sorted if c["shap_value"] > 0]

        # Mitigating factors (reducing default risk / negative SHAP)
        mitigating = [c for c in contributions_sorted if c["shap_value"] < 0]

        return {
            "index": idx,
            "base_value": base_val,
            "model_output": out_val,
            "reconstructed_output": float(base_val + np.sum(inst_shap)),
            "feature_contributions": contributions_sorted,
            "top_adverse_factors": adverse,
            "top_mitigating_factors": mitigating,
        }


class TreeSHAPExplainer:
    """
    High-Speed TreeSHAP Attribution Engine using C++ TreeExplainer.

    Supports:
    - LightGBM, XGBoost, CatBoost, Random Forest, Decision Tree
    - BaselineModel and CostSensitive*Classifier wrappers
    - Additivity verification: phi_0 + sum(phi_j(x)) == f(x)
    - Global feature importance & Local waterfall attributions
    - Pairwise TreeSHAP interaction computation
    """

    def __init__(
        self,
        model: Any,
        data: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        feature_names: Optional[List[str]] = None,
        model_output: str = "raw",
        feature_perturbation: str = "tree_path_dependent",
    ) -> None:
        """
        Initialize the TreeSHAPExplainer.

        Parameters:
        -----------
        model : Any
            Fitted tree-based model (native or Credex wrapper).
        data : Optional[Union[pd.DataFrame, np.ndarray]]
            Background dataset (optional). If None, uses tree_path_dependent exact TreeSHAP.
        feature_names : Optional[List[str]]
            Feature names (inferred from data or model if None).
        model_output : str, default="raw"
            SHAP output space ('raw', 'log_loss', 'probability').
        feature_perturbation : str, default="tree_path_dependent"
            Feature perturbation method for TreeExplainer.
        """
        self.raw_model = model
        self.underlying_estimator = extract_tree_estimator(model)
        self.feature_perturbation = feature_perturbation
        self.model_output = model_output

        # Infer feature names
        self.feature_names = feature_names
        if self.feature_names is None and hasattr(self.underlying_estimator, "feature_names_in_"):
            self.feature_names = list(self.underlying_estimator.feature_names_in_)
        elif self.feature_names is None and isinstance(data, pd.DataFrame):
            self.feature_names = list(data.columns)

        # Build native shap.TreeExplainer
        background = data.values if isinstance(data, pd.DataFrame) else data
        try:
            if background is not None:
                self.explainer = shap.TreeExplainer(
                    self.underlying_estimator,
                    data=background,
                    feature_perturbation="interventional",
                    model_output=self.model_output,
                )
            else:
                self.explainer = shap.TreeExplainer(
                    self.underlying_estimator,
                    feature_perturbation=self.feature_perturbation,
                    model_output=self.model_output,
                )
        except Exception as e:
            logger.warning(
                "Initial TreeExplainer initialization with %s raised: %s. Falling back to default.",
                self.feature_perturbation,
                e,
            )
            self.explainer = shap.TreeExplainer(self.underlying_estimator)

    def explain(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        max_samples: int = 5000,
        check_additivity: bool = True,
        tolerance: float = 1e-4,
    ) -> TreeSHAPResult:
        """
        Compute TreeSHAP attributions with additivity verification.

        Parameters:
        -----------
        X : Union[pd.DataFrame, np.ndarray]
            Dataset to explain.
        max_samples : int, default=5000
            Maximum number of samples to evaluate to ensure sub-minute runtimes.
        check_additivity : bool, default=True
            Whether to numerically verify the Shapley additivity property.
        tolerance : float, default=1e-4
            Maximum allowable discrepancy for additivity check.

        Returns:
        --------
        TreeSHAPResult
            Structured explanation container.
        """
        # Determine feature names
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
            X_df = X
            X_eval = X.iloc[:max_samples] if len(X) > max_samples else X
            data_arr = X_eval.values
        else:
            feature_names = (
                self.feature_names
                if self.feature_names is not None
                else [f"feature_{i}" for i in range(X.shape[1])]
            )
            X_df = pd.DataFrame(X, columns=feature_names)
            X_eval = X_df.iloc[:max_samples] if len(X) > max_samples else X_df
            data_arr = X_eval.values

        if len(X) > max_samples:
            logger.info(
                "Downsampled TreeSHAP evaluation from %d to %d samples for computational efficiency.",
                len(X),
                max_samples,
            )

        # Call TreeExplainer
        raw_explanation = self.explainer(X_eval)

        # Standardize to 2D explanation for positive class (class 1: default risk)
        raw_values = raw_explanation.values
        raw_base = raw_explanation.base_values

        if raw_values.ndim == 3 and raw_values.shape[2] >= 2:
            # Multi-class output (e.g. RandomForestClassifier) -> extract class 1
            shap_values_c1 = raw_values[:, :, 1]
            base_values_c1 = (
                raw_base[:, 1]
                if (isinstance(raw_base, np.ndarray) and raw_base.ndim > 1)
                else (
                    np.full(len(X_eval), raw_base[1])
                    if isinstance(raw_base, (list, tuple, np.ndarray)) and len(raw_base) == 2
                    else np.full(len(X_eval), raw_base)
                )
            )
            explanation = shap.Explanation(
                values=shap_values_c1,
                base_values=base_values_c1,
                data=data_arr,
                feature_names=feature_names,
            )
        else:
            shap_values_c1 = raw_values
            if np.isscalar(raw_base) or (isinstance(raw_base, np.ndarray) and raw_base.ndim == 0):
                base_values_c1 = np.full(len(X_eval), float(raw_base))
            elif isinstance(raw_base, np.ndarray) and raw_base.ndim == 1:
                if len(raw_base) == len(X_eval):
                    base_values_c1 = raw_base
                else:
                    base_values_c1 = np.full(len(X_eval), float(raw_base[0]))
            else:
                base_values_c1 = np.asarray(raw_base).ravel()

            explanation = shap.Explanation(
                values=shap_values_c1,
                base_values=base_values_c1,
                data=data_arr,
                feature_names=feature_names,
            )

        # Compute model output f(x)
        model_outputs = get_native_tree_output(self.underlying_estimator, X_eval)
        if model_outputs.ndim > 1 and model_outputs.shape[1] == 1:
            model_outputs = model_outputs.ravel()

        # Additivity verification: phi_0 + sum(phi_j) == f(x)
        reconstructed = base_values_c1 + np.sum(shap_values_c1, axis=1)
        abs_errors = np.abs(reconstructed - model_outputs)
        max_error = float(np.max(abs_errors))
        is_additive = bool(max_error <= tolerance)

        if check_additivity and not is_additive:
            logger.warning(
                "Shapley additivity check warning: max error %.2e exceeds tolerance %.2e",
                max_error,
                tolerance,
            )

        # Global feature importance: mean(|phi_j|)
        mean_abs_shap = np.mean(np.abs(shap_values_c1), axis=0)
        global_importance = pd.Series(mean_abs_shap, index=feature_names).sort_values(
            ascending=False
        )

        return TreeSHAPResult(
            explanation=explanation,
            shap_values=shap_values_c1,
            base_values=base_values_c1,
            feature_names=feature_names,
            data=data_arr,
            model_outputs=model_outputs,
            is_additive=is_additive,
            max_additivity_error=max_error,
            global_importance=global_importance,
        )

    def verify_additivity(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        tolerance: float = 1e-4,
    ) -> Tuple[bool, float, np.ndarray]:
        """
        Verify the Shapley additivity property phi_0 + sum_j(phi_j(x)) == f(x).

        Returns:
        --------
        Tuple[bool, float, np.ndarray]
            (is_additive, max_error, error_array)
        """
        result = self.explain(X, check_additivity=False)
        reconstructed = result.base_values + np.sum(result.shap_values, axis=1)
        abs_errors = np.abs(reconstructed - result.model_outputs)
        max_error = float(np.max(abs_errors))
        is_additive = bool(max_error <= tolerance)
        return is_additive, max_error, abs_errors

    def compute_interaction_values(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        max_samples: int = 1000,
    ) -> np.ndarray:
        """
        Compute pairwise TreeSHAP interaction values.

        Returns:
        --------
        np.ndarray
            Shape (N, M, M) interaction tensor where diagonal is main effects
            and off-diagonals are pairwise interaction attributions.
        """
        if isinstance(X, pd.DataFrame):
            X_eval = X.iloc[:max_samples] if len(X) > max_samples else X
        else:
            X_eval = X[:max_samples] if len(X) > max_samples else X

        try:
            interaction_vals = self.explainer.shap_interaction_values(X_eval)
            if interaction_vals.ndim == 4 and interaction_vals.shape[3] >= 2:
                # Class 1 interaction tensor
                return interaction_vals[:, :, :, 1]
            return interaction_vals
        except Exception as e:
            logger.error("Failed to compute TreeSHAP interaction values: %s", e)
            raise

    def explain_instance(
        self,
        x: Union[pd.Series, pd.DataFrame, np.ndarray],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Generate local explanation for a single applicant profile.
        """
        if isinstance(x, pd.Series):
            df_x = x.to_frame().T
        elif isinstance(x, np.ndarray) and x.ndim == 1:
            df_x = pd.DataFrame(
                [x],
                columns=self.feature_names
                or [f"feature_{i}" for i in range(len(x))],
            )
        elif isinstance(x, pd.DataFrame):
            df_x = x.iloc[[0]]
        else:
            df_x = pd.DataFrame(x)

        result = self.explain(df_x, max_samples=1)
        inst_dict = result.explain_instance(0)
        inst_dict["top_adverse_factors"] = inst_dict["top_adverse_factors"][:top_k]
        inst_dict["top_mitigating_factors"] = inst_dict["top_mitigating_factors"][:top_k]
        return inst_dict
