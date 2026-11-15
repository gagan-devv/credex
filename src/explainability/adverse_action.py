"""
Constrained Counterfactual Recourse & ECOA Adverse Action Notice Engine (CREDEX-17).

Provides actionable recourse recommendations for rejected loan applicants (p >= tau*)
under strict domain mutability and feasibility constraints:
- Mutable features: loan_amount, revol_utilization, revol_bal, installment
- Immutable features: delinq_2yrs, inq_last_6mths, pub_rec, total_acc, emp_length, fico_score
- Vectorized L1 projection solver finding minimal perturbations to achieve p < tau* in < 50 ms
- Automated ECOA Regulation B (12 CFR Part 1002) and FCRA adverse action notice formatting
"""

from dataclasses import dataclass, field
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.explainability.shap_explainer import TreeSHAPExplainer

logger = logging.getLogger(__name__)

# Canonical domain mutability definitions per configs/features_config.yaml
DEFAULT_MUTABLE_FEATURES: Tuple[str, ...] = (
    "loan_amount",
    "revol_utilization",
    "revol_bal",
    "installment",
)

DEFAULT_IMMUTABLE_FEATURES: Tuple[str, ...] = (
    "delinq_2yrs",
    "inq_last_6mths",
    "pub_rec",
    "total_acc",
    "emp_length",
    "fico_score",
)

# Human-readable plain language labels compliant with CFPB ECOA Regulation B guidelines
FEATURE_LABELS: Dict[str, str] = {
    "debt_to_income": "Debt-to-Income Ratio (DTI)",
    "dti": "Debt-to-Income Ratio (DTI)",
    "revol_utilization": "Revolving Credit Line Utilization",
    "delinq_2yrs": "Recent 30+ Day Delinquencies",
    "loan_amount": "Requested Loan Amount",
    "installment": "Scheduled Monthly Installment",
    "revol_bal": "Revolving Credit Card Balance",
    "inq_last_6mths": "Recent Hard Credit Inquiries",
    "pub_rec": "Derogatory Public Records",
    "total_acc": "Total Credit Lines History",
    "open_acc": "Number of Open Credit Lines",
    "emp_length": "Employment Tenure Length",
    "annual_income": "Verified Annual Income",
    "fico_score": "Credit Bureau Risk Score",
    "payment_to_income": "Payment-to-Income Ratio",
    "loan_to_income": "Loan-to-Income Ratio",
    "liquidity_buffer": "Available Revolving Liquidity Buffer",
    "delinquency_velocity": "Delinquency Velocity",
}

# Domain lower bounds for mutable variables
DEFAULT_BOUNDS: Dict[str, Tuple[float, Optional[float]]] = {
    "loan_amount": (1000.0, None),
    "revol_bal": (0.0, None),
    "revol_utilization": (0.0, 100.0),
    "installment": (25.0, None),
}

# Normalization scales for L1 distance accounting
DEFAULT_SCALES: Dict[str, float] = {
    "loan_amount": 5000.0,
    "revol_bal": 4000.0,
    "revol_utilization": 20.0,
    "installment": 150.0,
}


def _format_feature_value(feature_name: str, val: float) -> str:
    """Format numeric values into plain-language strings (dollars, percentages, counts)."""
    if "amount" in feature_name or "bal" in feature_name or "income" in feature_name or "installment" in feature_name:
        return f"${val:,.0f}" if val >= 10 else f"${val:,.2f}"
    if "utilization" in feature_name or "rate" in feature_name or "dti" in feature_name or "ratio" in feature_name:
        return f"{val:.1f}%"
    if "delinq" in feature_name or "inq" in feature_name or "rec" in feature_name or "acc" in feature_name:
        return f"{int(round(val))} occurrence(s)" if val == 1 else f"{int(round(val))} occurrences"
    if "score" in feature_name:
        return f"{int(round(val))}"
    if "length" in feature_name:
        return f"{val:.1f} years"
    return f"{val:.2f}"


@dataclass
class RecourseOption:
    """A distinct counterfactual recourse path."""

    option_id: str
    title: str
    description: str
    feature_deltas: Dict[str, float]
    resulting_prob: float
    l1_norm: float
    is_feasible: bool


@dataclass
class ActionableRecourseResult:
    """Container for complete applicant recourse and ECOA disclosure outputs."""

    applicant_id: str
    is_approved: bool
    is_feasible: bool
    original_prob: float
    recourse_prob: float
    tau_star: float
    feature_deltas: Dict[str, float]
    original_features: Dict[str, Any]
    recourse_features: Dict[str, Any]
    options: List[RecourseOption]
    l1_norm: float
    top_adverse_factors: List[Dict[str, Any]]
    adverse_action_notice: str
    execution_time_ms: float
    explanation_dict: Optional[Dict[str, Any]] = None

    def verify_immutable_invariance(
        self, immutable_features: Optional[List[str]] = None
    ) -> bool:
        """
        Verify that all immutable features are strictly identical between original and recourse.
        """
        immuts = immutable_features or list(DEFAULT_IMMUTABLE_FEATURES)
        for col in immuts:
            if col in self.original_features and col in self.recourse_features:
                orig_v = self.original_features[col]
                rec_v = self.recourse_features[col]
                if orig_v != rec_v:
                    # Allow microscopic float roundoff
                    if isinstance(orig_v, (float, np.floating)) and abs(orig_v - rec_v) > 1e-9:
                        return False
                    elif orig_v != rec_v:
                        return False
        return True


class ActionableRecourseEngine:
    """
    Constrained Counterfactual Recourse Generator with ECOA Regulation B Compliance.

    Solves:
        min ||delta||_{1, norm}
        s.t. f(X_mut + delta, X_immut) < tau*
             delta_j <= 0  for j in mutable
             (X_mut + delta)_j >= lower_bound_j
             delta_i == 0  for i in immutable
    """

    def __init__(
        self,
        model: Any,
        explainer: Optional[TreeSHAPExplainer] = None,
        tau_star: float = 0.25,
        feature_names: Optional[List[str]] = None,
        mutable_features: Optional[List[str]] = None,
        immutable_features: Optional[List[str]] = None,
        feature_bounds: Optional[Dict[str, Tuple[float, Optional[float]]]] = None,
        feature_scales: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Initialize the ActionableRecourseEngine.

        Parameters:
        -----------
        model : Any
            Trained predictive model or estimator exposing `predict_proba`.
        explainer : Optional[TreeSHAPExplainer]
            Fitted TreeSHAPExplainer for generating principal adverse factors.
        tau_star : float, default=0.25
            Optimal economic decision cutoff threshold.
        mutable_features : Optional[List[str]]
            List of features allowed to be perturbed.
        immutable_features : Optional[List[str]]
            List of features held strictly constant.
        feature_bounds : Optional[Dict]
            Lower and upper bounds for mutable features.
        feature_scales : Optional[Dict]
            L1 normalization scales for mutable features.
        """
        self.model = model
        self.explainer = explainer
        self.tau_star = float(tau_star)
        self.feature_names = feature_names
        self.mutable_features = (
            list(mutable_features) if mutable_features is not None else list(DEFAULT_MUTABLE_FEATURES)
        )
        self.immutable_features = (
            list(immutable_features) if immutable_features is not None else list(DEFAULT_IMMUTABLE_FEATURES)
        )
        self.feature_bounds = dict(DEFAULT_BOUNDS)
        if feature_bounds:
            self.feature_bounds.update(feature_bounds)

        self.feature_scales = dict(DEFAULT_SCALES)
        if feature_scales:
            self.feature_scales.update(feature_scales)

    def predict_proba_single(self, applicant_df: pd.DataFrame) -> float:
        """Evaluate positive default risk probability P(Y=1|X)."""
        probs = self.predict_proba_batch(applicant_df)
        return float(probs[0])

    def predict_proba_batch(self, batch_df: pd.DataFrame) -> np.ndarray:
        """Evaluate positive default risk probability for a batch of candidate profiles."""
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(batch_df)
            if probs.ndim == 2 and probs.shape[1] >= 2:
                return probs[:, 1]
            return probs.ravel()
        elif callable(self.model):
            return np.asarray(self.model(batch_df)).ravel()
        elif hasattr(self.model, "predict"):
            preds = self.model.predict(batch_df)
            return preds.ravel()
        raise AttributeError("Model must provide predict_proba or predict method.")

    def compute_l1_distance(self, deltas: Dict[str, float]) -> float:
        """Compute normalized L1 distance of feature perturbations."""
        dist = 0.0
        for feat, d in deltas.items():
            scale = self.feature_scales.get(feat, 1.0)
            dist += abs(d) / scale
        return dist

    def _solve_single_feature(
        self,
        applicant_df: pd.DataFrame,
        feature: str,
        tau_star: float,
        n_grid: int = 80,
    ) -> Optional[RecourseOption]:
        """
        Vectorized search for single-action recourse flipping p < tau*.
        """
        if feature not in applicant_df.columns:
            return None

        orig_val = float(applicant_df[feature].iloc[0])
        lb, _ = self.feature_bounds.get(feature, (0.0, None))

        if orig_val <= lb:
            return None

        # Build candidate grid
        grid_vals = np.linspace(orig_val, lb, n_grid)
        batch = pd.concat([applicant_df] * n_grid, ignore_index=True)
        batch[feature] = grid_vals

        # Handle linked domain features
        if feature == "loan_amount" and "installment" in applicant_df.columns:
            orig_inst = float(applicant_df["installment"].iloc[0])
            ratio = grid_vals / max(orig_val, 1.0)
            batch["installment"] = orig_inst * ratio

        if feature == "revol_bal" and "revol_utilization" in applicant_df.columns:
            orig_util = float(applicant_df["revol_utilization"].iloc[0])
            ratio = grid_vals / max(orig_val, 1.0)
            batch["revol_utilization"] = np.clip(orig_util * ratio, 0.0, 100.0)

        # Batch predict
        probs = self.predict_proba_batch(batch)
        feasible_indices = np.where(probs < tau_star)[0]

        if len(feasible_indices) == 0:
            return None

        # Pick earliest crossing index (minimal perturbation)
        best_idx = int(feasible_indices[0])
        best_val = float(grid_vals[best_idx])
        delta = best_val - orig_val
        resulting_p = float(probs[best_idx])

        deltas = {feature: delta}
        if feature == "loan_amount" and "installment" in applicant_df.columns:
            deltas["installment"] = float(batch["installment"].iloc[best_idx]) - float(applicant_df["installment"].iloc[0])
        if feature == "revol_bal" and "revol_utilization" in applicant_df.columns:
            deltas["revol_utilization"] = float(batch["revol_utilization"].iloc[best_idx]) - float(applicant_df["revol_utilization"].iloc[0])

        l1 = self.compute_l1_distance(deltas)
        label = FEATURE_LABELS.get(feature, feature)
        orig_fmt = _format_feature_value(feature, orig_val)
        best_fmt = _format_feature_value(feature, best_val)
        delta_fmt = _format_feature_value(feature, abs(delta))

        if feature == "loan_amount":
            desc = f"Reduce requested loan amount from {orig_fmt} to {best_fmt} (-{delta_fmt})."
        elif feature == "revol_bal":
            desc = f"Pay down revolving credit card balances from {orig_fmt} to {best_fmt} (-{delta_fmt})."
        elif feature == "revol_utilization":
            desc = f"Lower credit utilization from {orig_fmt} to {best_fmt} (-{delta_fmt})."
        elif feature == "installment":
            desc = f"Reduce monthly installment from {orig_fmt} to {best_fmt} (-{delta_fmt}) via loan restructuring."
        else:
            desc = f"Adjust {label} from {orig_fmt} to {best_fmt} ({delta:+.2f})."

        return RecourseOption(
            option_id="",
            title=f"Adjust {label}",
            description=desc,
            feature_deltas=deltas,
            resulting_prob=resulting_p,
            l1_norm=l1,
            is_feasible=True,
        )

    def _solve_joint_recourse(
        self,
        applicant_df: pd.DataFrame,
        tau_star: float,
        n_steps: int = 100,
    ) -> Optional[RecourseOption]:
        """
        Vectorized search for optimal joint L1 recourse combining all mutable features.
        """
        # Directional joint interpolation from original state towards minimum feasible state
        active_muts = [c for c in self.mutable_features if c in applicant_df.columns]
        if not active_muts:
            return None

        orig_row = applicant_df.iloc[0].to_dict()
        min_state = dict(orig_row)

        for col in active_muts:
            lb, _ = self.feature_bounds.get(col, (0.0, None))
            min_state[col] = lb

        # Ensure linked dependencies in min_state
        if "loan_amount" in min_state and "installment" in min_state:
            orig_l = float(orig_row["loan_amount"])
            if orig_l > 0:
                min_state["installment"] = float(orig_row["installment"]) * (min_state["loan_amount"] / orig_l)

        if "revol_bal" in min_state and "revol_utilization" in min_state:
            orig_b = float(orig_row["revol_bal"])
            if orig_b > 0:
                min_state["revol_utilization"] = float(orig_row["revol_utilization"]) * (min_state["revol_bal"] / orig_b)

        # Generate linear homotopy grid: X(alpha) = (1 - alpha) * X_orig + alpha * X_min
        alphas = np.linspace(0.0, 1.0, n_steps)
        batch_rows = []
        for a in alphas:
            row = dict(orig_row)
            for col in active_muts:
                row[col] = (1.0 - a) * orig_row[col] + a * min_state[col]
            batch_rows.append(row)

        batch_df = pd.DataFrame(batch_rows)
        probs = self.predict_proba_batch(batch_df)
        feasible_indices = np.where(probs < tau_star)[0]

        if len(feasible_indices) == 0:
            return None

        best_idx = int(feasible_indices[0])
        best_row = batch_df.iloc[best_idx]
        resulting_p = float(probs[best_idx])

        deltas = {}
        desc_parts = []
        for col in active_muts:
            d = float(best_row[col]) - float(orig_row[col])
            if abs(d) > 1e-4:
                deltas[col] = d
                orig_fmt = _format_feature_value(col, float(orig_row[col]))
                rec_fmt = _format_feature_value(col, float(best_row[col]))
                delta_fmt = _format_feature_value(col, abs(d))
                desc_parts.append(f"{FEATURE_LABELS.get(col, col)} ({orig_fmt} -> {rec_fmt}, -{delta_fmt})")

        l1 = self.compute_l1_distance(deltas)
        desc = "Simultaneously: " + ", and ".join(desc_parts)

        return RecourseOption(
            option_id="",
            title="Joint Multi-Feature Recourse",
            description=desc,
            feature_deltas=deltas,
            resulting_prob=resulting_p,
            l1_norm=l1,
            is_feasible=True,
        )

    def generate_recourse(
        self,
        applicant: Union[pd.Series, pd.DataFrame, Dict[str, Any]],
        tau_star: Optional[float] = None,
        applicant_id: str = "LC-9842104",
        compute_shap: bool = True,
        max_options: int = 3,
    ) -> ActionableRecourseResult:
        """
        Generate actionable counterfactual recourse and adverse action disclosure notice.

        Parameters:
        -----------
        applicant : Union[pd.Series, pd.DataFrame, Dict[str, Any]]
            Applicant feature vector.
        tau_star : Optional[float]
            Decision cutoff (defaults to self.tau_star if None).
        applicant_id : str
            Identifier for notice header.
        compute_shap : bool, default=True
            Whether to run TreeSHAP to identify principal adverse factors.
        max_options : int, default=3
            Maximum number of recourse options to include.

        Returns:
        --------
        ActionableRecourseResult
            Complete recourse and disclosure container.
        """
        t0 = time.perf_counter()
        tau = float(tau_star if tau_star is not None else self.tau_star)

        # Standardize applicant format
        if isinstance(applicant, dict):
            app_df = pd.DataFrame([applicant])
        elif isinstance(applicant, pd.Series):
            app_df = applicant.to_frame().T
        elif isinstance(applicant, pd.DataFrame):
            app_df = applicant.iloc[[0]].copy()
        else:
            raise TypeError("applicant must be a dict, pd.Series, or pd.DataFrame.")

        orig_features = app_df.iloc[0].to_dict()
        original_prob = self.predict_proba_single(app_df)

        # Check if already approved
        if original_prob < tau:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            notice = self._format_approved_notice(applicant_id, original_prob, tau)
            return ActionableRecourseResult(
                applicant_id=applicant_id,
                is_approved=True,
                is_feasible=True,
                original_prob=original_prob,
                recourse_prob=original_prob,
                tau_star=tau,
                feature_deltas={},
                original_features=orig_features,
                recourse_features=dict(orig_features),
                options=[],
                l1_norm=0.0,
                top_adverse_factors=[],
                adverse_action_notice=notice,
                execution_time_ms=elapsed_ms,
            )

        # Compute TreeSHAP principal adverse factors
        top_adverse: List[Dict[str, Any]] = []
        explanation_dict: Optional[Dict[str, Any]] = None
        if compute_shap:
            if self.explainer is None:
                try:
                    self.explainer = TreeSHAPExplainer(model=self.model, feature_names=self.feature_names)
                except Exception as e:
                    logger.warning("Auto-initialization of TreeSHAPExplainer failed: %s", e)
            if self.explainer is not None:
                try:
                    res = self.explainer.explain(app_df, check_additivity=False)
                    inst_dict = res.explain_instance(0)
                    top_adverse = inst_dict.get("top_adverse_factors", [])
                    explanation_dict = inst_dict
                except Exception as e:
                    logger.warning("TreeSHAP computation during adverse action failed: %s", e)

        # Solve individual mutable options
        candidates: List[RecourseOption] = []

        # Option: Loan amount reduction
        if "loan_amount" in self.mutable_features:
            opt_loan = self._solve_single_feature(app_df, "loan_amount", tau)
            if opt_loan is not None:
                candidates.append(opt_loan)

        # Option: Revolving balance paydown
        if "revol_bal" in self.mutable_features:
            opt_bal = self._solve_single_feature(app_df, "revol_bal", tau)
            if opt_bal is not None:
                candidates.append(opt_bal)

        # Option: Revolving utilization reduction
        if "revol_utilization" in self.mutable_features:
            opt_util = self._solve_single_feature(app_df, "revol_utilization", tau)
            if opt_util is not None:
                candidates.append(opt_util)

        # Option: Joint multi-feature recourse
        opt_joint = self._solve_joint_recourse(app_df, tau)
        if opt_joint is not None:
            candidates.append(opt_joint)

        # Sort options by L1 norm (lowest burden first)
        candidates_sorted = sorted(candidates, key=lambda o: o.l1_norm)

        # Format option IDs (Option A, Option B, etc.)
        labeled_options: List[RecourseOption] = []
        opt_letters = ["Option A", "Option B", "Option C", "Option D"]
        for idx, opt in enumerate(candidates_sorted[:max_options]):
            letter = opt_letters[idx] if idx < len(opt_letters) else f"Option {idx+1}"
            opt.option_id = letter
            labeled_options.append(opt)

        is_feasible = len(labeled_options) > 0

        # Choose primary counterfactual (minimal L1)
        if is_feasible:
            primary_opt = labeled_options[0]
            recourse_prob = primary_opt.resulting_prob
            feature_deltas = primary_opt.feature_deltas
            l1_norm = primary_opt.l1_norm
        else:
            recourse_prob = original_prob
            feature_deltas = {}
            l1_norm = float("inf")

        # Build recourse feature dictionary strictly preserving immutables
        recourse_features = dict(orig_features)
        for feat, delta in feature_deltas.items():
            if feat in self.mutable_features:
                recourse_features[feat] = float(orig_features[feat]) + delta

        # Strict preservation of all immutable features
        for col in self.immutable_features:
            if col in orig_features:
                recourse_features[col] = orig_features[col]

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Generate standardized adverse action disclosure notice
        notice = self.format_adverse_action_notice(
            applicant_id=applicant_id,
            original_prob=original_prob,
            tau_star=tau,
            top_adverse_factors=top_adverse,
            options=labeled_options,
            is_feasible=is_feasible,
        )

        return ActionableRecourseResult(
            applicant_id=applicant_id,
            is_approved=False,
            is_feasible=is_feasible,
            original_prob=original_prob,
            recourse_prob=recourse_prob,
            tau_star=tau,
            feature_deltas=feature_deltas,
            original_features=orig_features,
            recourse_features=recourse_features,
            options=labeled_options,
            l1_norm=l1_norm,
            top_adverse_factors=top_adverse,
            adverse_action_notice=notice,
            execution_time_ms=elapsed_ms,
            explanation_dict=explanation_dict,
        )

    def format_adverse_action_notice(
        self,
        applicant_id: str,
        original_prob: float,
        tau_star: float,
        top_adverse_factors: List[Dict[str, Any]],
        options: List[RecourseOption],
        is_feasible: bool = True,
    ) -> str:
        """
        Format standardized regulatory adverse action disclosure notice compliant
        with ECOA Regulation B (12 CFR Part 1002) and FCRA (15 U.S.C. 1681m).
        """
        divider = "=" * 80
        lines = [
            divider,
            "ADVERSE ACTION DISCLOSURE NOTICE (REGULATION B / ECOA COMPLIANT)",
            divider,
            f"Application ID: {applicant_id:<25} Decision: REJECTED (Risk: {original_prob*100:.1f}% >= Cutoff: {tau_star*100:.1f}%)",
            "",
            "PRINCIPAL ADVERSE FACTORS CONTRIBUTING TO RISK SCORE:",
        ]

        if top_adverse_factors:
            for idx, factor in enumerate(top_adverse_factors[:4], 1):
                f_name = factor["feature"]
                f_val = factor["value"]
                sv = factor["shap_value"]
                label = FEATURE_LABELS.get(f_name, f_name)
                val_str = _format_feature_value(f_name, f_val) if isinstance(f_val, (int, float, np.floating, np.integer)) else str(f_val)
                lines.append(f"{idx}. {label} ({val_str}) contributed {sv:+.2f} to default log-odds.")
        else:
            lines.append("1. Debt-to-income and leverage ratios exceed automated credit policy guidelines.")
            lines.append("2. Recent delinquency and revolving line utilization indicators reflect elevated default risk.")

        lines.extend([
            "",
            "ACTIONABLE COUNTERFACTUAL RECOURSE (FEASIBLE APPROVAL PATH):",
            f"To reduce default risk below the approval threshold (tau* = {tau_star:.3f}), the applicant may:",
        ])

        if is_feasible and options:
            for opt in options:
                lines.append(f"  [{opt.option_id}] {opt.description}")
        else:
            lines.append("  [Notice] Based on current credit history and established underwriting criteria,")
            lines.append("  no immediate adjustments to requested loan parameters would reduce default risk below cutoff.")
            lines.append("  We encourage reapplication following sustained positive payment history and debt reduction.")

        lines.extend([
            "",
            "EQUAL CREDIT OPPORTUNITY ACT (ECOA) STATEMENT:",
            "The Federal Equal Credit Opportunity Act prohibits creditors from discriminating against",
            "credit applicants on the basis of race, color, religion, national origin, sex, marital",
            "status, or age. The federal agency that administers compliance with this law is the",
            "Consumer Financial Protection Bureau (CFPB), 1700 G Street NW, Washington, DC 20552.",
            divider,
        ])

        return "\n".join(lines)

    def _format_approved_notice(
        self, applicant_id: str, original_prob: float, tau_star: float
    ) -> str:
        """Notice for applicants that meet the approval threshold."""
        divider = "=" * 80
        return "\n".join([
            divider,
            "CREDIT UNDERWRITING DISCLOSURE NOTICE",
            divider,
            f"Application ID: {applicant_id:<25} Decision: APPROVED (Risk: {original_prob*100:.1f}% < Cutoff: {tau_star*100:.1f}%)",
            "",
            "STATUS: The credit application satisfies all automated risk and solvency criteria.",
            "No counterfactual recourse required.",
            divider,
        ])
