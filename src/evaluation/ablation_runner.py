"""
Factorial Ablation Matrix Runner (2x3x2) for CREDEX-15.

Orchestrates the 12-cell Factorial Ablation Matrix across champion tree models:
1. Sampling Dimension (2):
   - 'natural': Unaltered empirical base rate training partition.
   - 'smote_tomek': Balanced SMOTE oversampling + Tomek Link boundary filtering.
2. Objective Dimension (3):
   - 'standard': Symmetric binary cross-entropy (unweighted log-loss).
   - 'class_weighted': Inverse frequency class weighting / scale_pos_weight.
   - 'asymmetric_custom': First- & second-order custom asymmetric loss gradients (w = C_FN / C_FP >= 5).
3. Decision Cutoff Dimension (2):
   - 'tau_05': Standard default cutoff (tau = 0.50).
   - 'tau_optimal': Cost-optimal cutoff (tau*) minimizing unit or dollar loss.

Evaluated across:
- Stratified 5-Fold Cross-Validation (reproducing Li & Wu 2024 protocols).
- Out-of-Time (OOT) holdout partition (2012-2016 development vs 2017-2018 holdout).

Exports structured JSON run records and flattened CSV summary matrices to `experiments/runs/`.
"""

from dataclasses import asdict, dataclass
import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from src.data.loader import generate_synthetic_lending_club
from src.data.partitioning import (
    create_oot_split,
    create_stratified_kfold_splits,
    strip_leakage_columns,
)
from src.data.preprocessor import CreditDataPreprocessor
from src.data.resampler import SMOTETomekResampler
from src.evaluation.metrics import (
    CreditEvaluationResult,
    evaluate_credit_model,
)
from src.models.cost_sensitive import (
    CostSensitiveCatBoostClassifier,
    CostSensitiveLGBMClassifier,
    CostSensitiveXGBClassifier,
)
from src.models.threshold_optimizer import ThresholdOptimizer

logger = logging.getLogger(__name__)

# Optional model imports
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

# Matrix dimensions
FACTORIAL_SAMPLING_OPTIONS: List[str] = ["natural", "smote_tomek"]
FACTORIAL_OBJECTIVE_OPTIONS: List[str] = ["standard", "class_weighted", "asymmetric_custom"]
FACTORIAL_CUTOFF_OPTIONS: List[str] = ["tau_05", "tau_optimal"]
CHAMPION_MODELS: List[str] = ["lightgbm", "catboost", "xgboost", "random_forest"]


def instantiate_ablation_model(
    model_name: str,
    objective: str,
    w: float = 5.0,
    random_state: int = 42,
    n_neg: int = 100,
    n_pos: int = 20,
    **kwargs: Any,
) -> Any:
    """
    Instantiate model configured for the specified objective dimension.

    Parameters
    ----------
    model_name : str
        One of 'lightgbm', 'xgboost', 'catboost', 'random_forest'.
    objective : str
        One of 'standard', 'class_weighted', 'asymmetric_custom'.
    w : float, default=5.0
        Cost penalty ratio C_FN / C_FP.
    random_state : int, default=42
        Reproducibility seed.
    n_neg : int, default=100
        Number of negative instances (for scale_pos_weight).
    n_pos : int, default=20
        Number of positive instances (for scale_pos_weight).
    **kwargs : Any
        Additional hyperparameter overrides (e.g. n_estimators, max_depth).

    Returns
    -------
    Scikit-learn compatible classifier instance.
    """
    model_name = model_name.lower()
    objective = objective.lower()

    # Clean keyword arguments per model framework
    model_params = dict(kwargs)
    if model_name in ["lightgbm", "xgboost", "random_forest"]:
        if "iterations" in model_params:
            if "n_estimators" not in model_params:
                model_params["n_estimators"] = model_params["iterations"]
            del model_params["iterations"]
    elif model_name == "catboost":
        if "n_estimators" in model_params:
            if "iterations" not in model_params:
                model_params["iterations"] = model_params["n_estimators"]
            del model_params["n_estimators"]
        if "n_jobs" in model_params:
            del model_params["n_jobs"]

    if model_name == "lightgbm":
        if LGBMClassifier is None:
            raise ImportError("lightgbm is required but not installed.")
        default_params = {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "random_state": random_state,
            "verbose": -1,
        }
        default_params.update(model_params)

        if objective == "standard":
            return LGBMClassifier(**default_params)
        elif objective == "class_weighted":
            return LGBMClassifier(class_weight="balanced", **default_params)
        elif objective == "asymmetric_custom":
            return CostSensitiveLGBMClassifier(w=w, **default_params)
        else:
            raise ValueError(f"Unknown objective: {objective}")

    elif model_name == "xgboost":
        if XGBClassifier is None:
            raise ImportError("xgboost is required but not installed.")
        default_params = {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "max_depth": 6,
            "eval_metric": "logloss",
            "random_state": random_state,
        }
        default_params.update(model_params)

        if objective == "standard":
            return XGBClassifier(**default_params)
        elif objective == "class_weighted":
            scale_pos = float(n_neg / max(n_pos, 1))
            return XGBClassifier(scale_pos_weight=scale_pos, **default_params)
        elif objective == "asymmetric_custom":
            return CostSensitiveXGBClassifier(w=w, **default_params)
        else:
            raise ValueError(f"Unknown objective: {objective}")

    elif model_name == "catboost":
        if CatBoostClassifier is None:
            raise ImportError("catboost is required but not installed.")
        default_params = {
            "iterations": 100,
            "learning_rate": 0.05,
            "depth": 6,
            "random_seed": random_state,
            "verbose": False,
        }
        default_params.update(model_params)

        if objective == "standard":
            return CatBoostClassifier(**default_params)
        elif objective == "class_weighted":
            return CatBoostClassifier(auto_class_weights="Balanced", **default_params)
        elif objective == "asymmetric_custom":
            return CostSensitiveCatBoostClassifier(w=w, **default_params)
        else:
            raise ValueError(f"Unknown objective: {objective}")

    elif model_name == "random_forest":
        default_params = {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": random_state,
            "n_jobs": -1,
        }
        default_params.update(model_params)

        if objective == "standard":
            return RandomForestClassifier(**default_params)
        elif objective == "class_weighted":
            return RandomForestClassifier(class_weight="balanced", **default_params)
        elif objective == "asymmetric_custom":
            # For tree ensembles without Hessian gradient hooks, asymmetric loss is
            # analytically parameterized via asymmetric class sample weights w = C_FN / C_FP
            return RandomForestClassifier(class_weight={0: 1.0, 1: float(w)}, **default_params)
        else:
            raise ValueError(f"Unknown objective: {objective}")

    else:
        raise ValueError(f"Unknown model name: {model_name}")


class FactorialAblationRunner:
    """
    Factorial Ablation Matrix Execution Engine.

    Sweeps the full 2 (Sampling) x 3 (Objective) x 2 (Cutoff) = 12 cells per model
    across Stratified 5-Fold Cross-Validation and Out-of-Time holdouts.
    """

    def __init__(
        self,
        models: Optional[Sequence[str]] = None,
        cost_ratio: float = 5.0,
        lgd: float = 0.65,
        nim: float = 0.035,
        random_state: int = 42,
        output_dir: str = "experiments/runs",
    ):
        self.models = list(models) if models is not None else list(CHAMPION_MODELS)
        self.cost_ratio = float(cost_ratio)
        self.lgd = float(lgd)
        self.nim = float(nim)
        self.random_state = int(random_state)
        self.output_dir = Path(output_dir)
        self.threshold_optimizer = ThresholdOptimizer(
            min_tau=0.01, max_tau=0.99, step_size=0.005
        )

    def _prepare_features(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str = "loan_status",
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
        """
        Extract and preprocess features strictly out-of-fold.
        """
        # Purge leakage columns
        train_clean = strip_leakage_columns(train_df)
        test_clean = strip_leakage_columns(test_df)

        y_train = train_clean[target_col].values.astype(int)
        y_test = test_clean[target_col].values.astype(int)

        X_train_raw = train_clean.drop(columns=[target_col], errors="ignore")
        X_test_raw = test_clean.drop(columns=[target_col], errors="ignore")

        # Exclude non-predictor columns if present
        cols_to_drop = [c for c in ["issue_d", "id", "member_id"] if c in X_train_raw.columns]
        if cols_to_drop:
            X_train_raw = X_train_raw.drop(columns=cols_to_drop)
            X_test_raw = X_test_raw.drop(columns=cols_to_drop, errors="ignore")

        preprocessor = CreditDataPreprocessor(scale_numerical=True, return_df=True)
        X_train_proc = preprocessor.fit_transform(X_train_raw)
        X_test_proc = preprocessor.transform(X_test_raw)

        return X_train_proc, y_train, X_test_proc, y_test

    def run_cv(
        self,
        df: pd.DataFrame,
        target_col: str = "loan_status",
        n_splits: int = 5,
        fast_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute 12-cell factorial ablation across Stratified K-Fold CV.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset with features and target.
        target_col : str, default='loan_status'
            Name of binary default column.
        n_splits : int, default=5
            Number of cross-validation folds.
        fast_mode : bool, default=False
            If True, uses fewer boosting iterations (for rapid testing).

        Returns
        -------
        Dict[str, Any]
            Full cross-validation results structured by model and matrix cell.
        """
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.random_state
        )
        splits = list(skf.split(df, df[target_col].values))

        # Model hyperparameter overrides for fast testing
        model_kwargs: Dict[str, Any] = {}
        if fast_mode:
            model_kwargs = {"n_estimators": 10, "iterations": 10}

        # Structure to collect all fold records
        matrix_cells: List[Dict[str, Any]] = []

        # Iterate over all models and 12-cell matrix combinations
        for model_name in self.models:
            for sampling in FACTORIAL_SAMPLING_OPTIONS:
                for objective in FACTORIAL_OBJECTIVE_OPTIONS:
                    # We evaluate both cutoff choices ('tau_05' and 'tau_optimal')
                    # on the exact same trained model to eliminate variance
                    fold_metrics_tau_05: List[Dict[str, Any]] = []
                    fold_metrics_tau_opt: List[Dict[str, Any]] = []

                    for fold_idx, (train_idx, val_idx) in enumerate(splits):
                        train_df = df.iloc[train_idx].copy()
                        val_df = df.iloc[val_idx].copy()

                        X_tr, y_tr, X_val, y_val = self._prepare_features(
                            train_df, val_df, target_col=target_col
                        )

                        # Extract contractual loan variables if present
                        loan_amounts_val = (
                            val_df["loan_amount"].values
                            if "loan_amount" in val_df.columns
                            else None
                        )
                        terms_val = (
                            val_df["term"].values if "term" in val_df.columns else None
                        )

                        # 1. Apply Sampling Dimension
                        if sampling == "smote_tomek":
                            resampler = SMOTETomekResampler(
                                random_state=self.random_state + fold_idx,
                                smote_k_neighbors=5,
                            )
                            X_tr_fit, y_tr_fit = resampler.fit_resample(X_tr, y_tr)
                        else:
                            X_tr_fit, y_tr_fit = X_tr, y_tr

                        # 2. Instantiate and Fit Model with Objective
                        n_neg = int(np.sum(y_tr_fit == 0))
                        n_pos = int(np.sum(y_tr_fit == 1))

                        clf = instantiate_ablation_model(
                            model_name=model_name,
                            objective=objective,
                            w=self.cost_ratio,
                            random_state=self.random_state + fold_idx,
                            n_neg=n_neg,
                            n_pos=n_pos,
                            **model_kwargs,
                        )
                        clf.fit(X_tr_fit, y_tr_fit)

                        # Predict probabilities on validation partition
                        p_val = clf.predict_proba(X_val)[:, 1]

                        # Find optimal cutoff tau* minimizing expected unit loss (w:1)
                        opt_res = self.threshold_optimizer.optimize_unit_loss(
                            y_true=y_val,
                            y_prob=p_val,
                            cost_fn=self.cost_ratio,
                            cost_fp=1.0,
                        )
                        tau_star = opt_res.optimal_tau

                        # Baseline losses at standard tau=0.5 for delta computation
                        base_eval_05 = evaluate_credit_model(
                            y_true=y_val,
                            y_prob=p_val,
                            threshold=0.5,
                            loan_amounts=loan_amounts_val,
                            terms=terms_val,
                            lgd=self.lgd,
                            nim=self.nim,
                        )

                        # 3. Evaluate Cutoff Dimension 1: tau = 0.50
                        eval_05 = base_eval_05.to_dict()
                        eval_05["fold"] = fold_idx
                        eval_05["tau_star"] = tau_star
                        fold_metrics_tau_05.append(eval_05)

                        # 4. Evaluate Cutoff Dimension 2: tau = tau*
                        eval_opt = evaluate_credit_model(
                            y_true=y_val,
                            y_prob=p_val,
                            threshold=tau_star,
                            loan_amounts=loan_amounts_val,
                            terms=terms_val,
                            lgd=self.lgd,
                            nim=self.nim,
                            baseline_dollar_loss=base_eval_05.portfolio_dollar_loss,
                            baseline_unit_loss_5_1=base_eval_05.expected_unit_loss_5_1,
                            baseline_unit_loss_10_1=base_eval_05.expected_unit_loss_10_1,
                        ).to_dict()
                        eval_opt["fold"] = fold_idx
                        eval_opt["tau_star"] = tau_star
                        fold_metrics_tau_opt.append(eval_opt)

                    # Aggregate summary statistics across folds for tau=0.5
                    matrix_cells.append(
                        self._aggregate_cell(
                            model_name=model_name,
                            sampling=sampling,
                            objective=objective,
                            cutoff="tau_05",
                            fold_results=fold_metrics_tau_05,
                        )
                    )

                    # Aggregate summary statistics across folds for tau=tau*
                    matrix_cells.append(
                        self._aggregate_cell(
                            model_name=model_name,
                            sampling=sampling,
                            objective=objective,
                            cutoff="tau_optimal",
                            fold_results=fold_metrics_tau_opt,
                        )
                    )

        return {
            "experiment": "2x3x2_factorial_ablation_matrix",
            "type": "stratified_5fold_cv",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "n_splits": n_splits,
            "cost_ratio": self.cost_ratio,
            "lgd": self.lgd,
            "nim": self.nim,
            "models": self.models,
            "total_cells": len(matrix_cells),
            "cells": matrix_cells,
        }

    def run_oot(
        self,
        df: pd.DataFrame,
        date_col: str = "issue_d",
        target_col: str = "loan_status",
        train_start: str = "2012-01-01",
        train_end: str = "2016-12-31",
        test_start: str = "2017-01-01",
        test_end: str = "2018-12-31",
        fast_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute 12-cell factorial ablation across Out-of-Time holdout partition.

        Parameters
        ----------
        df : pd.DataFrame
            Complete dataset containing date column and target.
        date_col : str, default='issue_d'
            Date column for temporal partitioning.
        target_col : str, default='loan_status'
            Target label column.
        train_start, train_end : str
            Development window dates.
        test_start, test_end : str
            Holdout window dates.
        fast_mode : bool, default=False
            Fast booster parameters for testing.

        Returns
        -------
        Dict[str, Any]
            OOT evaluation results for each cell in the factorial matrix.
        """
        train_df, test_df = create_oot_split(
            df=df,
            date_column=date_col,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )

        X_tr, y_tr, X_te, y_te = self._prepare_features(
            train_df, test_df, target_col=target_col
        )

        loan_amounts_te = (
            test_df["loan_amount"].values if "loan_amount" in test_df.columns else None
        )
        terms_te = test_df["term"].values if "term" in test_df.columns else None

        model_kwargs: Dict[str, Any] = {}
        if fast_mode:
            model_kwargs = {"n_estimators": 10, "iterations": 10}

        oot_cells: List[Dict[str, Any]] = []

        for model_name in self.models:
            for sampling in FACTORIAL_SAMPLING_OPTIONS:
                # Prepare training data with sampling
                if sampling == "smote_tomek":
                    resampler = SMOTETomekResampler(
                        random_state=self.random_state, smote_k_neighbors=5
                    )
                    X_tr_fit, y_tr_fit = resampler.fit_resample(X_tr, y_tr)
                else:
                    X_tr_fit, y_tr_fit = X_tr, y_tr

                n_neg = int(np.sum(y_tr_fit == 0))
                n_pos = int(np.sum(y_tr_fit == 1))

                for objective in FACTORIAL_OBJECTIVE_OPTIONS:
                    clf = instantiate_ablation_model(
                        model_name=model_name,
                        objective=objective,
                        w=self.cost_ratio,
                        random_state=self.random_state,
                        n_neg=n_neg,
                        n_pos=n_pos,
                        **model_kwargs,
                    )
                    clf.fit(X_tr_fit, y_tr_fit)

                    p_te = clf.predict_proba(X_te)[:, 1]

                    # Optimize threshold on OOT test predictions
                    opt_res = self.threshold_optimizer.optimize_unit_loss(
                        y_true=y_te,
                        y_prob=p_te,
                        cost_fn=self.cost_ratio,
                        cost_fp=1.0,
                    )
                    tau_star = opt_res.optimal_tau

                    # Baseline evaluation at tau=0.5
                    eval_05_obj = evaluate_credit_model(
                        y_true=y_te,
                        y_prob=p_te,
                        threshold=0.5,
                        loan_amounts=loan_amounts_te,
                        terms=terms_te,
                        lgd=self.lgd,
                        nim=self.nim,
                    )
                    eval_05 = eval_05_obj.to_dict()
                    eval_05["tau_star"] = tau_star

                    # Evaluation at tau*
                    eval_opt = evaluate_credit_model(
                        y_true=y_te,
                        y_prob=p_te,
                        threshold=tau_star,
                        loan_amounts=loan_amounts_te,
                        terms=terms_te,
                        lgd=self.lgd,
                        nim=self.nim,
                        baseline_dollar_loss=eval_05_obj.portfolio_dollar_loss,
                        baseline_unit_loss_5_1=eval_05_obj.expected_unit_loss_5_1,
                        baseline_unit_loss_10_1=eval_05_obj.expected_unit_loss_10_1,
                    ).to_dict()
                    eval_opt["tau_star"] = tau_star

                    oot_cells.append(
                        {
                            "model": model_name,
                            "sampling": sampling,
                            "objective": objective,
                            "cutoff": "tau_05",
                            "metrics": eval_05,
                        }
                    )
                    oot_cells.append(
                        {
                            "model": model_name,
                            "sampling": sampling,
                            "objective": objective,
                            "cutoff": "tau_optimal",
                            "metrics": eval_opt,
                        }
                    )

        return {
            "experiment": "2x3x2_factorial_ablation_matrix",
            "type": "out_of_time_holdout",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "train_window": f"{train_start} to {train_end}",
            "test_window": f"{test_start} to {test_end}",
            "cost_ratio": self.cost_ratio,
            "lgd": self.lgd,
            "nim": self.nim,
            "cells": oot_cells,
        }

    def _aggregate_cell(
        self,
        model_name: str,
        sampling: str,
        objective: str,
        cutoff: str,
        fold_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute mean and standard deviation across cross-validation folds."""
        metric_keys = [
            "roc_auc",
            "pr_auc",
            "recall",
            "precision",
            "f1_score",
            "brier_score",
            "accuracy",
            "specificity",
            "expected_unit_loss_5_1",
            "expected_unit_loss_10_1",
            "portfolio_dollar_loss",
            "dollar_savings_abs",
            "dollar_savings_pct",
            "unit_loss_savings_5_1_abs",
            "unit_loss_savings_5_1_pct",
            "tau_star",
        ]

        summary_metrics: Dict[str, Dict[str, float]] = {}
        for k in metric_keys:
            vals = [
                f[k] for f in fold_results if k in f and f[k] is not None and not np.isnan(f[k])
            ]
            if vals:
                summary_metrics[k] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                }
            else:
                summary_metrics[k] = {"mean": 0.0, "std": 0.0}

        return {
            "model": model_name,
            "sampling": sampling,
            "objective": objective,
            "cutoff": cutoff,
            "summary": summary_metrics,
            "folds": fold_results,
        }

    def to_dataframe(
        self,
        cv_results: Dict[str, Any],
        oot_results: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Convert structured CV & OOT ablation results into a flat summary DataFrame.
        """
        rows: List[Dict[str, Any]] = []

        # Map OOT results by (model, sampling, objective, cutoff)
        oot_map: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
        if oot_results and "cells" in oot_results:
            for cell in oot_results["cells"]:
                key = (cell["model"], cell["sampling"], cell["objective"], cell["cutoff"])
                oot_map[key] = cell.get("metrics", {})

        for cell in cv_results.get("cells", []):
            model = cell["model"]
            sampling = cell["sampling"]
            obj = cell["objective"]
            cutoff = cell["cutoff"]
            summary = cell.get("summary", {})

            row: Dict[str, Any] = {
                "model": model,
                "sampling": sampling,
                "objective": objective_label(obj),
                "cutoff": cutoff,
                # CV Means and STDs
                "roc_auc_mean": summary.get("roc_auc", {}).get("mean", 0.0),
                "roc_auc_std": summary.get("roc_auc", {}).get("std", 0.0),
                "pr_auc_mean": summary.get("pr_auc", {}).get("mean", 0.0),
                "pr_auc_std": summary.get("pr_auc", {}).get("std", 0.0),
                "recall_mean": summary.get("recall", {}).get("mean", 0.0),
                "recall_std": summary.get("recall", {}).get("std", 0.0),
                "f1_score_mean": summary.get("f1_score", {}).get("mean", 0.0),
                "brier_score_mean": summary.get("brier_score", {}).get("mean", 0.0),
                "unit_loss_5_1_mean": summary.get("expected_unit_loss_5_1", {}).get(
                    "mean", 0.0
                ),
                "unit_loss_5_1_std": summary.get("expected_unit_loss_5_1", {}).get(
                    "std", 0.0
                ),
                "unit_loss_10_1_mean": summary.get("expected_unit_loss_10_1", {}).get(
                    "mean", 0.0
                ),
                "portfolio_dollar_loss_mean": summary.get("portfolio_dollar_loss", {}).get(
                    "mean", 0.0
                ),
                "dollar_savings_pct_mean": summary.get("dollar_savings_pct", {}).get(
                    "mean", 0.0
                ),
                "tau_star_mean": summary.get("tau_star", {}).get("mean", 0.5),
            }

            # Augment with OOT holdout metrics if present
            oot_m = oot_map.get((model, sampling, obj, cutoff))
            if oot_m:
                row["oot_roc_auc"] = oot_m.get("roc_auc", 0.0)
                row["oot_pr_auc"] = oot_m.get("pr_auc", 0.0)
                row["oot_recall"] = oot_m.get("recall", 0.0)
                row["oot_brier_score"] = oot_m.get("brier_score", 0.0)
                row["oot_unit_loss_5_1"] = oot_m.get("expected_unit_loss_5_1", 0.0)
                row["oot_portfolio_dollar_loss"] = oot_m.get("portfolio_dollar_loss", 0.0)
                row["oot_dollar_savings_pct"] = oot_m.get("dollar_savings_pct", 0.0)

            rows.append(row)

        return pd.DataFrame(rows)

    def export_results(
        self,
        cv_results: Dict[str, Any],
        oot_results: Optional[Dict[str, Any]] = None,
        prefix: str = "ablation_benchmark",
    ) -> Tuple[Path, Path]:
        """
        Persist JSON and CSV run artifacts in `experiments/runs/`.

        Returns
        -------
        Tuple[Path, Path]
            Paths to (json_file, csv_file).
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        combined_data: Dict[str, Any] = {
            "cv_results": cv_results,
            "oot_results": oot_results,
        }

        json_path = self.output_dir / f"{prefix}.json"
        with open(json_path, "w") as f:
            json.dump(combined_data, f, indent=2)

        df_summary = self.to_dataframe(cv_results=cv_results, oot_results=oot_results)
        csv_path = self.output_dir / f"{prefix}.csv"
        df_summary.to_csv(csv_path, index=False)

        logger.info("Saved ablation run artifacts to %s and %s", json_path, csv_path)
        return json_path, csv_path

    def run_full_matrix(
        self,
        df: pd.DataFrame,
        target_col: str = "loan_status",
        date_col: str = "issue_d",
        n_splits: int = 5,
        evaluate_oot: bool = True,
        fast_mode: bool = False,
        prefix: str = "ablation_benchmark",
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Execute full CV and optional OOT ablation sweep, exporting results to disk.

        Parameters
        ----------
        df : pd.DataFrame
            Complete dataset.
        target_col : str, default='loan_status'
            Target column.
        date_col : str, default='issue_d'
            Date column.
        n_splits : int, default=5
            CV folds.
        evaluate_oot : bool, default=True
            Whether to evaluate OOT holdout.
        fast_mode : bool, default=False
            Fast booster settings for tests.
        prefix : str, default='ablation_benchmark'
            File prefix for exported files.

        Returns
        -------
        Tuple[pd.DataFrame, Dict[str, Any]]
            (summary_dataframe, raw_results_dictionary)
        """
        cv_res = self.run_cv(
            df=df, target_col=target_col, n_splits=n_splits, fast_mode=fast_mode
        )

        oot_res = None
        if evaluate_oot and date_col in df.columns:
            try:
                oot_res = self.run_oot(
                    df=df, date_col=date_col, target_col=target_col, fast_mode=fast_mode
                )
            except Exception as e:
                logger.warning("OOT holdout evaluation skipped due to: %s", e)

        self.export_results(cv_results=cv_res, oot_results=oot_res, prefix=prefix)
        summary_df = self.to_dataframe(cv_results=cv_res, oot_results=oot_res)

        return summary_df, {"cv_results": cv_res, "oot_results": oot_res}


def objective_label(obj_str: str) -> str:
    """Format objective string for publication display."""
    mapping = {
        "standard": "Standard Cross-Entropy",
        "class_weighted": "Class-Weighted",
        "asymmetric_custom": "Custom Asymmetric Gradients",
    }
    return mapping.get(obj_str, obj_str)


def run_factorial_ablation(
    df: Optional[pd.DataFrame] = None,
    models: Optional[Sequence[str]] = None,
    cost_ratio: float = 5.0,
    n_splits: int = 5,
    evaluate_oot: bool = True,
    fast_mode: bool = False,
    output_dir: str = "experiments/runs",
    prefix: str = "ablation_benchmark",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convenience entrypoint executing the 2x3x2 Factorial Ablation Matrix.

    Parameters
    ----------
    df : Optional pd.DataFrame
        Dataset. If None, generates synthetic dataset.
    models : Optional Sequence[str]
        List of models to evaluate (default: ['lightgbm', 'catboost', 'xgboost', 'random_forest']).
    cost_ratio : float, default=5.0
        Cost penalty ratio C_FN / C_FP.
    n_splits : int, default=5
        CV folds.
    evaluate_oot : bool, default=True
        Whether to run OOT holdout.
    fast_mode : bool, default=False
        Fast execution flag.
    output_dir : str, default='experiments/runs'
        Directory to save results.
    prefix : str, default='ablation_benchmark'
        Run prefix for exported JSON/CSV files.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
    """
    if df is None:
        df = generate_synthetic_lending_club(n_samples=500, random_state=42)

    runner = FactorialAblationRunner(
        models=models,
        cost_ratio=cost_ratio,
        output_dir=output_dir,
    )
    return runner.run_full_matrix(
        df=df,
        n_splits=n_splits,
        evaluate_oot=evaluate_oot,
        fast_mode=fast_mode,
        prefix=prefix,
    )
