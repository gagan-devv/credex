"""
Comprehensive Financial & Statistical Evaluation Suite (CREDEX-14).

Implements end-to-end evaluation calculating both:
1. Statistical Discrimination Metrics:
   - ROC-AUC: Area under the Receiver Operating Characteristic curve.
   - PR-AUC: Area under the Precision-Recall curve (Average Precision).
   - Default Recall: True Positive Rate (Sensitivity) at decision threshold tau.
   - Precision: Positive Predictive Value at decision threshold tau.
   - F1-Score: Harmonic mean of precision and recall.
   - Brier Score: Mean squared calibration error strictly in [0, 1].
   - Accuracy: Overall classification accuracy.
   - Specificity: True Negative Rate (TN / (TN + FP)).

2. Financial Economic Balance-Sheet Metrics:
   - Normalized Expected Unit Loss under 5:1 (benchmark) and 10:1 (stress scenario):
     Loss_unit = (w * FN + FP) / N
   - Total Portfolio Dollar Loss under contractual exposures:
     Loss_portfolio = sum_{i in FN} (loan_amount_i * LGD) + sum_{j in FP} (loan_amount_j * NIM * term_j / 12)
   - Dollar Savings (Delta $ and % savings) vs Li & Wu (2024) baseline model / tau=0.5 cutoff:
     Delta $ = Loss_baseline - Loss_model
     % Savings = (Delta $ / Loss_baseline) * 100
"""

from dataclasses import asdict, dataclass
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models.threshold_optimizer import (
    compute_confusion_at_thresholds,
    compute_portfolio_dollar_loss,
    compute_unit_expected_loss,
    parse_term_in_months,
)

logger = logging.getLogger(__name__)


@dataclass
class CreditEvaluationResult:
    """Structured container for all statistical and financial evaluation metrics."""

    # Statistical discrimination metrics
    roc_auc: float
    pr_auc: float
    recall: float
    precision: float
    f1_score: float
    brier_score: float
    accuracy: float
    specificity: float

    # Financial economic metrics
    expected_unit_loss_5_1: float
    expected_unit_loss_10_1: float
    portfolio_dollar_loss: Optional[float] = None

    # Economic savings vs baseline
    dollar_savings_abs: Optional[float] = None
    dollar_savings_pct: Optional[float] = None
    unit_loss_savings_5_1_abs: Optional[float] = None
    unit_loss_savings_5_1_pct: Optional[float] = None
    unit_loss_savings_10_1_abs: Optional[float] = None
    unit_loss_savings_10_1_pct: Optional[float] = None

    # Confusion matrix and execution metadata
    threshold: float = 0.5
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    n_samples: int = 0
    lgd: float = 0.65
    nim: float = 0.035
    baseline_portfolio_loss: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a standard serializable dictionary."""
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, (np.floating, np.integer)):
                d[k] = float(v) if isinstance(v, np.floating) else int(v)
        return d

    def to_series(self) -> pd.Series:
        """Convert metrics to a pandas Series."""
        return pd.Series(self.to_dict())


def compute_brier_score(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    y_prob: Union[Sequence[float], np.ndarray, pd.Series],
) -> float:
    """
    Compute Brier Score (Mean Squared Probability Calibration Error).

    Always strictly bounded in [0, 1].

    Parameters
    ----------
    y_true : array-like
        Binary ground truth labels {0, 1}.
    y_prob : array-like
        Predicted probabilities for positive class (default).

    Returns
    -------
    float
        Brier score in [0, 1].
    """
    y = np.asarray(y_true, dtype=np.int32).ravel()
    p = np.asarray(y_prob, dtype=np.float64).ravel()
    if len(y) != len(p):
        raise ValueError(f"Length mismatch between y_true ({len(y)}) and y_prob ({len(p)})")
    if len(y) == 0:
        return 0.0
    # Ensure probabilities clipped to [0, 1]
    p_clipped = np.clip(p, 0.0, 1.0)
    score = float(brier_score_loss(y, p_clipped))
    return float(np.clip(score, 0.0, 1.0))


def compute_statistical_metrics(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    y_prob: Union[Sequence[float], np.ndarray, pd.Series],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute full statistical discrimination suite.

    Parameters
    ----------
    y_true : array-like
        Binary true labels.
    y_prob : array-like
        Predicted default probabilities.
    threshold : float, default=0.5
        Decision cutoff for binary prediction.

    Returns
    -------
    Dict[str, float]
        Dictionary with roc_auc, pr_auc, recall, precision, f1_score,
        brier_score, accuracy, specificity.
    """
    y = np.asarray(y_true, dtype=np.int32).ravel()
    p = np.asarray(y_prob, dtype=np.float64).ravel()
    p_clipped = np.clip(p, 0.0, 1.0)
    y_pred = (p_clipped >= threshold).astype(np.int32)

    unique_classes = np.unique(y)
    if len(unique_classes) > 1:
        roc_auc = float(roc_auc_score(y, p_clipped))
        pr_auc = float(average_precision_score(y, p_clipped))
    else:
        roc_auc = 0.5
        pr_auc = float(np.mean(y)) if len(y) > 0 else 0.0

    brier = compute_brier_score(y, p_clipped)
    acc = float(accuracy_score(y, y_pred))
    rec = float(recall_score(y, y_pred, zero_division=0.0))
    prec = float(precision_score(y, y_pred, zero_division=0.0))
    f1 = float(f1_score(y, y_pred, zero_division=0.0))

    # Specificity: TN / (TN + FP)
    tn = int(np.sum((y == 0) & (y_pred == 0)))
    fp = int(np.sum((y == 0) & (y_pred == 1)))
    denom_spec = tn + fp
    spec = float(tn / denom_spec) if denom_spec > 0 else 0.0

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "recall": rec,
        "precision": prec,
        "f1_score": f1,
        "brier_score": brier,
        "accuracy": acc,
        "specificity": spec,
    }


def compute_expected_unit_loss(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    y_prob: Union[Sequence[float], np.ndarray, pd.Series],
    threshold: float = 0.5,
    cost_fn: float = 5.0,
    cost_fp: float = 1.0,
    normalize: bool = True,
) -> float:
    """
    Calculate Expected Unit Misclassification Loss at a given threshold.

    Loss = (cost_fn * FN + cost_fp * FP) / (N if normalize else 1)

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_prob : array-like
        Predicted probabilities.
    threshold : float, default=0.5
        Decision threshold.
    cost_fn : float, default=5.0
        Cost of false negative (defaulting loan classified as approved).
    cost_fp : float, default=1.0
        Cost of false positive (creditworthy applicant rejected).
    normalize : bool, default=True
        Whether to divide by N.

    Returns
    -------
    float
        Normalized or total expected unit loss.
    """
    losses, _, _ = compute_unit_expected_loss(
        y_true=y_true,
        y_prob=y_prob,
        thresholds=np.array([threshold]),
        cost_fn=cost_fn,
        cost_fp=cost_fp,
        normalize=normalize,
    )
    return float(losses[0])


def compute_portfolio_loss(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    y_prob: Union[Sequence[float], np.ndarray, pd.Series],
    loan_amounts: Union[Sequence[float], np.ndarray, pd.Series],
    terms: Union[Sequence[Any], np.ndarray, pd.Series],
    threshold: float = 0.5,
    lgd: float = 0.65,
    nim: float = 0.035,
) -> float:
    """
    Calculate Total Portfolio Dollar Loss at a given decision threshold.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_prob : array-like
        Predicted default probabilities.
    loan_amounts : array-like
        Contract loan amounts in dollars.
    terms : array-like
        Contract loan terms (months).
    threshold : float, default=0.5
        Decision cutoff.
    lgd : float, default=0.65
        Loss Given Default.
    nim : float, default=0.035
        Net Interest Margin.

    Returns
    -------
    float
        Total portfolio dollar loss.
    """
    losses, _, _ = compute_portfolio_dollar_loss(
        y_true=y_true,
        y_prob=y_prob,
        loan_amounts=loan_amounts,
        terms=terms,
        thresholds=np.array([threshold]),
        lgd=lgd,
        nim=nim,
    )
    return float(losses[0])


def compute_dollar_savings(
    current_loss: float,
    baseline_loss: float,
) -> Tuple[float, float]:
    """
    Calculate absolute delta $ and percentage dollar savings vs baseline.

    Parameters
    ----------
    current_loss : float
        Dollar loss of the proposed model or optimal cutoff.
    baseline_loss : float
        Dollar loss of the baseline model or standard tau=0.5 cutoff.

    Returns
    -------
    Tuple[float, float]
        (delta_dollar_savings, percentage_savings)
    """
    delta_abs = float(baseline_loss - current_loss)
    if baseline_loss > 1e-12:
        pct = float((delta_abs / baseline_loss) * 100.0)
    else:
        pct = 0.0
    return delta_abs, pct


def evaluate_credit_model(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    y_prob: Union[Sequence[float], np.ndarray, pd.Series],
    threshold: float = 0.5,
    loan_amounts: Optional[Union[Sequence[float], np.ndarray, pd.Series]] = None,
    terms: Optional[Union[Sequence[Any], np.ndarray, pd.Series]] = None,
    lgd: float = 0.65,
    nim: float = 0.035,
    baseline_prob: Optional[Union[Sequence[float], np.ndarray, pd.Series]] = None,
    baseline_dollar_loss: Optional[float] = None,
    baseline_unit_loss_5_1: Optional[float] = None,
    baseline_unit_loss_10_1: Optional[float] = None,
) -> CreditEvaluationResult:
    """
    Full-pass evaluation suite computing all statistical and financial metrics.

    Parameters
    ----------
    y_true : array-like
        Binary ground truth labels {0, 1}.
    y_prob : array-like
        Predicted probabilities for default.
    threshold : float, default=0.5
        Classification decision threshold tau.
    loan_amounts : Optional array-like
        Loan amounts for portfolio dollar calculation.
    terms : Optional array-like
        Loan terms for opportunity cost calculation.
    lgd : float, default=0.65
        Loss Given Default parameter.
    nim : float, default=0.035
        Net Interest Margin parameter.
    baseline_prob : Optional array-like
        Probabilities of baseline model (e.g., Li & Wu replication at tau=0.5).
        If provided, computes explicit dollar and unit savings vs this baseline.
    baseline_dollar_loss : Optional float
        Pre-computed baseline dollar loss.
    baseline_unit_loss_5_1 : Optional float
        Pre-computed baseline 5:1 unit loss.
    baseline_unit_loss_10_1 : Optional float
        Pre-computed baseline 10:1 unit loss.

    Returns
    -------
    CreditEvaluationResult
    """
    y = np.asarray(y_true, dtype=np.int32).ravel()
    p = np.asarray(y_prob, dtype=np.float64).ravel()
    n_samples = len(y)

    # 1. Statistical discrimination metrics
    stats = compute_statistical_metrics(y, p, threshold=threshold)

    # Confusion matrix counts at threshold
    conf = compute_confusion_at_thresholds(y, p, np.array([threshold]))
    tp = int(conf["tp"][0])
    fp = int(conf["fp"][0])
    tn = int(conf["tn"][0])
    fn = int(conf["fn"][0])

    # 2. Financial unit loss metrics
    unit_5_1 = compute_expected_unit_loss(
        y, p, threshold=threshold, cost_fn=5.0, cost_fp=1.0, normalize=True
    )
    unit_10_1 = compute_expected_unit_loss(
        y, p, threshold=threshold, cost_fn=10.0, cost_fp=1.0, normalize=True
    )

    # 3. Financial portfolio dollar loss
    portfolio_loss: Optional[float] = None
    if loan_amounts is not None and terms is not None:
        portfolio_loss = compute_portfolio_loss(
            y_true=y,
            y_prob=p,
            loan_amounts=loan_amounts,
            terms=terms,
            threshold=threshold,
            lgd=lgd,
            nim=nim,
        )

    # 4. Baseline savings comparison
    # Compute baseline losses if baseline_prob is supplied
    if baseline_prob is not None:
        p_base = np.asarray(baseline_prob, dtype=np.float64).ravel()
        if baseline_unit_loss_5_1 is None:
            baseline_unit_loss_5_1 = compute_expected_unit_loss(
                y, p_base, threshold=0.5, cost_fn=5.0, cost_fp=1.0, normalize=True
            )
        if baseline_unit_loss_10_1 is None:
            baseline_unit_loss_10_1 = compute_expected_unit_loss(
                y, p_base, threshold=0.5, cost_fn=10.0, cost_fp=1.0, normalize=True
            )
        if baseline_dollar_loss is None and loan_amounts is not None and terms is not None:
            baseline_dollar_loss = compute_portfolio_loss(
                y_true=y,
                y_prob=p_base,
                loan_amounts=loan_amounts,
                terms=terms,
                threshold=0.5,
                lgd=lgd,
                nim=nim,
            )
    elif baseline_dollar_loss is None and threshold != 0.5:
        # Default fallback baseline: evaluate current model at standard tau=0.5
        if loan_amounts is not None and terms is not None:
            baseline_dollar_loss = compute_portfolio_loss(
                y_true=y,
                y_prob=p,
                loan_amounts=loan_amounts,
                terms=terms,
                threshold=0.5,
                lgd=lgd,
                nim=nim,
            )
        if baseline_unit_loss_5_1 is None:
            baseline_unit_loss_5_1 = compute_expected_unit_loss(
                y, p, threshold=0.5, cost_fn=5.0, cost_fp=1.0, normalize=True
            )
        if baseline_unit_loss_10_1 is None:
            baseline_unit_loss_10_1 = compute_expected_unit_loss(
                y, p, threshold=0.5, cost_fn=10.0, cost_fp=1.0, normalize=True
            )

    # Calculate savings metrics
    dollar_savings_abs: Optional[float] = None
    dollar_savings_pct: Optional[float] = None
    if portfolio_loss is not None and baseline_dollar_loss is not None:
        dollar_savings_abs, dollar_savings_pct = compute_dollar_savings(
            portfolio_loss, baseline_dollar_loss
        )

    u5_abs: Optional[float] = None
    u5_pct: Optional[float] = None
    if baseline_unit_loss_5_1 is not None:
        u5_abs, u5_pct = compute_dollar_savings(unit_5_1, baseline_unit_loss_5_1)

    u10_abs: Optional[float] = None
    u10_pct: Optional[float] = None
    if baseline_unit_loss_10_1 is not None:
        u10_abs, u10_pct = compute_dollar_savings(unit_10_1, baseline_unit_loss_10_1)

    return CreditEvaluationResult(
        roc_auc=stats["roc_auc"],
        pr_auc=stats["pr_auc"],
        recall=stats["recall"],
        precision=stats["precision"],
        f1_score=stats["f1_score"],
        brier_score=stats["brier_score"],
        accuracy=stats["accuracy"],
        specificity=stats["specificity"],
        expected_unit_loss_5_1=unit_5_1,
        expected_unit_loss_10_1=unit_10_1,
        portfolio_dollar_loss=portfolio_loss,
        dollar_savings_abs=dollar_savings_abs,
        dollar_savings_pct=dollar_savings_pct,
        unit_loss_savings_5_1_abs=u5_abs,
        unit_loss_savings_5_1_pct=u5_pct,
        unit_loss_savings_10_1_abs=u10_abs,
        unit_loss_savings_10_1_pct=u10_pct,
        threshold=float(threshold),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        n_samples=n_samples,
        lgd=float(lgd),
        nim=float(nim),
        baseline_portfolio_loss=baseline_dollar_loss,
    )


def evaluate_lgd_sensitivity(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    y_prob: Union[Sequence[float], np.ndarray, pd.Series],
    loan_amounts: Union[Sequence[float], np.ndarray, pd.Series],
    terms: Union[Sequence[Any], np.ndarray, pd.Series],
    threshold: float = 0.5,
    baseline_threshold: float = 0.5,
    baseline_prob: Optional[Union[Sequence[float], np.ndarray, pd.Series]] = None,
    lgd_scenarios: Optional[List[float]] = None,
    nim: float = 0.035,
) -> pd.DataFrame:
    """
    Evaluate Portfolio Dollar Loss across multiple LGD sensitivity scenarios (e.g. 60%, 70%, 75%).

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_prob : array-like
        Predicted probabilities.
    loan_amounts : array-like
        Loan amounts in dollars.
    terms : array-like
        Loan terms in months.
    threshold : float, default=0.5
        Model evaluation threshold (e.g. tau*).
    baseline_threshold : float, default=0.5
        Baseline threshold (usually tau=0.5).
    baseline_prob : Optional array-like
        Probabilities of baseline model.
    lgd_scenarios : Optional List[float]
        LGD values to evaluate (default: [0.60, 0.65, 0.70, 0.75]).
    nim : float, default=0.035
        Net Interest Margin.

    Returns
    -------
    pd.DataFrame
        Table with LGD scenario, model dollar loss, baseline dollar loss,
        delta dollar savings, and percentage savings.
    """
    if lgd_scenarios is None:
        lgd_scenarios = [0.60, 0.65, 0.70, 0.75]

    base_p = y_prob if baseline_prob is None else baseline_prob

    records = []
    for lgd_val in lgd_scenarios:
        loss_model = compute_portfolio_loss(
            y_true=y_true,
            y_prob=y_prob,
            loan_amounts=loan_amounts,
            terms=terms,
            threshold=threshold,
            lgd=lgd_val,
            nim=nim,
        )
        loss_base = compute_portfolio_loss(
            y_true=y_true,
            y_prob=base_p,
            loan_amounts=loan_amounts,
            terms=terms,
            threshold=baseline_threshold,
            lgd=lgd_val,
            nim=nim,
        )
        delta_abs, delta_pct = compute_dollar_savings(loss_model, loss_base)
        records.append(
            {
                "lgd": float(lgd_val),
                "model_loss": float(loss_model),
                "baseline_loss": float(loss_base),
                "dollar_savings_abs": float(delta_abs),
                "dollar_savings_pct": float(delta_pct),
            }
        )

    return pd.DataFrame(records)
