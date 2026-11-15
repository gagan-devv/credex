"""
Economic Loss Decision Threshold Optimizer (CREDEX-12).

This module implements vectorized decision threshold optimization across tau in [0.01, 0.99]
to find argmin E[Loss(tau)] under two financial evaluation criteria:
1. Standardized Unit Cost Matrix (C_FN : C_FP = 5:1, 10:1):
       Loss_unit(tau) = C_FN * FN(tau) + C_FP * FP(tau)
2. Instance-Dependent Portfolio Dollar Loss:
       Loss_portfolio(tau) = sum_{i in FN(tau)} (loan_amount_i * LGD)
                           + sum_{j in FP(tau)} (loan_amount_j * NIM * term_j / 12)

Key Properties:
- Optimal tau* under C_FN:C_FP = 5:1 is strictly less than 0.50 (typically in [0.15, 0.30]).
- As w = C_FN / C_FP increases, optimal tau* strictly monotonically decreases
  (higher risk aversion).
- Losses are non-negative.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class ThresholdOptimizationResult:
    """Structured container for threshold optimization results."""

    optimal_tau: float
    min_loss: float
    baseline_loss_tau_05: float
    loss_reduction_absolute: float
    loss_reduction_pct: float
    thresholds: np.ndarray
    losses: np.ndarray
    fn_counts: np.ndarray
    fp_counts: np.ndarray
    loss_type: str
    cost_ratio: Optional[float] = None
    lgd: Optional[float] = None
    nim: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert scalar summary metrics to a dictionary."""
        return {
            "optimal_tau": float(self.optimal_tau),
            "min_loss": float(self.min_loss),
            "baseline_loss_tau_05": float(self.baseline_loss_tau_05),
            "loss_reduction_absolute": float(self.loss_reduction_absolute),
            "loss_reduction_pct": float(self.loss_reduction_pct),
            "loss_type": self.loss_type,
            "cost_ratio": self.cost_ratio,
            "lgd": self.lgd,
            "nim": self.nim,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Export threshold sweep curve to a pandas DataFrame."""
        return pd.DataFrame(
            {
                "threshold": self.thresholds,
                "loss": self.losses,
                "fn_count": self.fn_counts,
                "fp_count": self.fp_counts,
            }
        )


def parse_term_in_months(terms: Union[Sequence[Any], np.ndarray, pd.Series]) -> np.ndarray:
    """
    Parse loan term into numeric months (e.g., '36 months' -> 36.0).

    Parameters
    ----------
    terms : array-like
        Loan terms as integers, floats, or strings (e.g. '36 months', ' 60 months').

    Returns
    -------
    np.ndarray
        Array of floats representing terms in months.
    """
    if isinstance(terms, pd.Series):
        if pd.api.types.is_numeric_dtype(terms):
            return terms.fillna(36.0).astype(np.float64).values
        terms_list = terms.tolist()
    elif isinstance(terms, np.ndarray):
        if np.issubdtype(terms.dtype, np.number):
            return terms.astype(np.float64)
        terms_list = terms.tolist()
    else:
        terms_list = list(terms)

    # String / mixed parsing
    result = np.empty(len(terms_list), dtype=np.float64)
    for i, t in enumerate(terms_list):
        if t is None or (isinstance(t, float) and np.isnan(t)) or pd.isna(t):
            result[i] = 36.0  # default LendingClub term
        elif isinstance(t, (int, float, np.integer, np.floating)):
            result[i] = float(t)
        else:
            digits = "".join(ch for ch in str(t) if ch.isdigit())
            result[i] = float(digits) if digits else 36.0
    return result


def compute_confusion_at_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Vectorized computation of TP, FP, TN, FN across an array of thresholds.

    Decision rule:
        prediction = (p >= tau)
        y = 1 is default (bad loan), y = 0 is non-default (good loan)
        FN: y=1 and p < tau (approving a defaulting loan)
        FP: y=0 and p >= tau (rejecting a creditworthy loan)

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground truth labels (0 or 1).
    y_prob : np.ndarray
        Predicted default probabilities in [0, 1].
    thresholds : np.ndarray
        Candidate thresholds tau in (0, 1).

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with arrays of counts ('tp', 'fp', 'tn', 'fn') for each threshold.
    """
    y = np.asarray(y_true, dtype=np.int32).ravel()
    p = np.asarray(y_prob, dtype=np.float64).ravel()
    taus = np.asarray(thresholds, dtype=np.float64).ravel()

    # Mask of positive predictions: shape (N, M)
    pred_pos = p[:, None] >= taus[None, :]

    y_pos = (y == 1)[:, None]
    y_neg = (y == 0)[:, None]

    tp = np.sum(pred_pos & y_pos, axis=0)
    fp = np.sum(pred_pos & y_neg, axis=0)
    fn = np.sum((~pred_pos) & y_pos, axis=0)
    tn = np.sum((~pred_pos) & y_neg, axis=0)

    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def compute_unit_expected_loss(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
    cost_fn: float = 5.0,
    cost_fp: float = 1.0,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized computation of expected unit loss across thresholds.

        Loss(tau) = C_FN * FN(tau) + C_FP * FP(tau)

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels (0 or 1).
    y_prob : np.ndarray
        Predicted default probabilities.
    thresholds : np.ndarray
        Threshold sweep values.
    cost_fn : float, default=5.0
        Cost of False Negative (approving a defaulting loan).
    cost_fp : float, default=1.0
        Cost of False Positive (rejecting a creditworthy loan).
    normalize : bool, default=True
        Whether to divide total loss by total sample count N.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (losses, fn_counts, fp_counts)
    """
    conf = compute_confusion_at_thresholds(y_true, y_prob, thresholds)
    fn = conf["fn"]
    fp = conf["fp"]

    total_loss = cost_fn * fn + cost_fp * fp
    if normalize:
        n_samples = len(y_true)
        losses = total_loss / max(n_samples, 1)
    else:
        losses = total_loss.astype(np.float64)

    return losses, fn, fp


def compute_portfolio_dollar_loss(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    loan_amounts: np.ndarray,
    terms: np.ndarray,
    thresholds: np.ndarray,
    lgd: float = 0.65,
    nim: float = 0.035,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized calculation of total portfolio credit loss in dollars across thresholds:

        Loss_portfolio(tau) = sum_{i in FN(tau)} (loan_amount_i * LGD)
                            + sum_{j in FP(tau)} (loan_amount_j * NIM * term_j / 12)

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_prob : np.ndarray
        Predicted probabilities in [0, 1].
    loan_amounts : np.ndarray
        Principal loan amount requested ($).
    terms : np.ndarray
        Loan term in months (e.g., 36, 60).
    thresholds : np.ndarray
        Candidate thresholds tau in (0, 1).
    lgd : float, default=0.65
        Loss Given Default rate (0.60 to 0.75).
    nim : float, default=0.035
        Annualized Net Interest Margin (3.5%).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (dollar_losses, fn_counts, fp_counts)
    """
    y = np.asarray(y_true, dtype=np.int32).ravel()
    p = np.asarray(y_prob, dtype=np.float64).ravel()
    amounts = np.asarray(loan_amounts, dtype=np.float64).ravel()
    terms_m = parse_term_in_months(terms)
    taus = np.asarray(thresholds, dtype=np.float64).ravel()

    # Per-instance cost vectors
    # Default write-off cost for positives (y=1)
    cost_fn_i = amounts * lgd
    # Foregone interest opportunity cost for negatives (y=0)
    cost_fp_j = amounts * nim * (terms_m / 12.0)

    # Indicator masks: shape (N, M)
    pred_pos = p[:, None] >= taus[None, :]

    fn_mask = (~pred_pos) & ((y == 1)[:, None])
    fp_mask = pred_pos & ((y == 0)[:, None])

    # Vectorized weighted sum across instances
    fn_dollar_loss = np.sum(fn_mask * cost_fn_i[:, None], axis=0)
    fp_dollar_loss = np.sum(fp_mask * cost_fp_j[:, None], axis=0)

    total_dollar_loss = fn_dollar_loss + fp_dollar_loss
    fn_counts = np.sum(fn_mask, axis=0)
    fp_counts = np.sum(fp_mask, axis=0)

    return total_dollar_loss, fn_counts, fp_counts


class ThresholdOptimizer:
    """
    Economic Loss Threshold Optimizer.

    Sweeps decision threshold tau in [min_tau, max_tau] to find the cost-optimal
    cutoff tau* minimizing expected unit loss or portfolio dollar loss.
    """

    def __init__(
        self,
        min_tau: float = 0.01,
        max_tau: float = 0.99,
        step_size: float = 0.005,
    ):
        self.min_tau = float(min_tau)
        self.max_tau = float(max_tau)
        self.step_size = float(step_size)
        self.thresholds = np.arange(self.min_tau, self.max_tau + 1e-9, self.step_size)

    def optimize_unit_loss(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        cost_fn: float = 5.0,
        cost_fp: float = 1.0,
        normalize: bool = True,
    ) -> ThresholdOptimizationResult:
        """
        Find optimal threshold tau* minimizing expected unit loss.

        Parameters
        ----------
        y_true : np.ndarray
            True binary labels.
        y_prob : np.ndarray
            Predicted probabilities.
        cost_fn : float, default=5.0
            Cost of False Negative.
        cost_fp : float, default=1.0
            Cost of False Positive.
        normalize : bool, default=True
            Normalize by N.

        Returns
        -------
        ThresholdOptimizationResult
        """
        losses, fn_counts, fp_counts = compute_unit_expected_loss(
            y_true=y_true,
            y_prob=y_prob,
            thresholds=self.thresholds,
            cost_fn=cost_fn,
            cost_fp=cost_fp,
            normalize=normalize,
        )

        best_idx = int(np.argmin(losses))
        optimal_tau = float(self.thresholds[best_idx])
        min_loss = float(losses[best_idx])

        # Baseline at tau = 0.50
        baseline_idx = int(np.argmin(np.abs(self.thresholds - 0.50)))
        baseline_loss = float(losses[baseline_idx])

        reduction_abs = baseline_loss - min_loss
        reduction_pct = (reduction_abs / max(baseline_loss, 1e-12)) * 100.0

        return ThresholdOptimizationResult(
            optimal_tau=optimal_tau,
            min_loss=min_loss,
            baseline_loss_tau_05=baseline_loss,
            loss_reduction_absolute=reduction_abs,
            loss_reduction_pct=reduction_pct,
            thresholds=self.thresholds,
            losses=losses,
            fn_counts=fn_counts,
            fp_counts=fp_counts,
            loss_type="unit",
            cost_ratio=float(cost_fn / cost_fp),
        )

    def optimize_portfolio_dollar_loss(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        loan_amounts: np.ndarray,
        terms: np.ndarray,
        lgd: float = 0.65,
        nim: float = 0.035,
    ) -> ThresholdOptimizationResult:
        """
        Find optimal threshold tau* minimizing total portfolio dollar loss.

        Parameters
        ----------
        y_true : np.ndarray
            True binary labels.
        y_prob : np.ndarray
            Predicted probabilities.
        loan_amounts : np.ndarray
            Loan amounts in dollars.
        terms : np.ndarray
            Loan terms (e.g. '36 months' or 36).
        lgd : float, default=0.65
            Loss Given Default.
        nim : float, default=0.035
            Net Interest Margin.

        Returns
        -------
        ThresholdOptimizationResult
        """
        dollar_losses, fn_counts, fp_counts = compute_portfolio_dollar_loss(
            y_true=y_true,
            y_prob=y_prob,
            loan_amounts=loan_amounts,
            terms=terms,
            thresholds=self.thresholds,
            lgd=lgd,
            nim=nim,
        )

        best_idx = int(np.argmin(dollar_losses))
        optimal_tau = float(self.thresholds[best_idx])
        min_loss = float(dollar_losses[best_idx])

        # Baseline at tau = 0.50
        baseline_idx = int(np.argmin(np.abs(self.thresholds - 0.50)))
        baseline_loss = float(dollar_losses[baseline_idx])

        reduction_abs = baseline_loss - min_loss
        reduction_pct = (reduction_abs / max(baseline_loss, 1e-12)) * 100.0

        return ThresholdOptimizationResult(
            optimal_tau=optimal_tau,
            min_loss=min_loss,
            baseline_loss_tau_05=baseline_loss,
            loss_reduction_absolute=reduction_abs,
            loss_reduction_pct=reduction_pct,
            thresholds=self.thresholds,
            losses=dollar_losses,
            fn_counts=fn_counts,
            fp_counts=fp_counts,
            loss_type="portfolio_dollar",
            lgd=float(lgd),
            nim=float(nim),
        )
