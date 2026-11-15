"""
Imbalance Remediation Engine using SMOTE-Tomek Hybrid Resampling (CREDEX-13).

Combines Synthetic Minority Over-sampling Technique (SMOTE) with Tomek Link
boundary noise removal:
1. SMOTE interpolates synthetic examples along minority class k-nearest neighbor line segments.
2. Tomek links detect and remove ambiguous, borderline pairs of opposite classes to clarify
   the decision boundary.

Strict Leakage Invariant:
Resampling is applied strictly to training partitions and NEVER to validation or
test partitions, ensuring that empirical test base rates and risk distributions
remain uncorrupted.
"""

import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks

logger = logging.getLogger(__name__)


class SMOTETomekResampler:
    """
    SMOTE-Tomek Hybrid Resampling Pipeline.

    Parameters
    ----------
    sampling_strategy : float, str, or callable, default="auto"
        Sampling strategy for SMOTE. "auto" balances minority to match majority.
    random_state : int, default=42
        Seed for reproducibility.
    smote_k_neighbors : int, default=5
        Number of nearest neighbors to used to construct synthetic samples.
    n_jobs : Optional[int], default=None
        Number of CPU cores to use.
    """

    def __init__(
        self,
        sampling_strategy: Union[float, str] = "auto",
        random_state: int = 42,
        smote_k_neighbors: int = 5,
        n_jobs: Optional[int] = None,
    ):
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.smote_k_neighbors = smote_k_neighbors
        self.n_jobs = n_jobs
        self.diagnostics_: Optional[Dict[str, Any]] = None

    def fit_resample(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]:
        """
        Resample dataset using SMOTE followed by Tomek Link elimination.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray of shape (n_samples, n_features)
            Predictor feature matrix.
        y : pd.Series or np.ndarray of shape (n_samples,)
            Binary target labels {0, 1}.

        Returns
        -------
        X_resampled : pd.DataFrame or np.ndarray
            Resampled feature matrix.
        y_resampled : pd.Series or np.ndarray
            Resampled binary labels.
        """
        is_df = isinstance(X, pd.DataFrame)
        is_series = isinstance(y, pd.Series)

        # Retain column names and series name
        columns = X.columns.tolist() if is_df else None
        target_name = y.name if is_series else "target"

        y_arr = np.asarray(y)
        if y_arr.ndim != 1:
            y_arr = y_arr.ravel()

        unique_classes, counts = np.unique(y_arr, return_counts=True)
        if len(unique_classes) != 2:
            raise ValueError(
                f"SMOTETomek requires binary target labels, found classes {unique_classes}"
            )

        n_majority = int(np.max(counts))
        n_minority = int(np.min(counts))

        if n_minority < 2:
            raise ValueError(
                f"Minority class has only {n_minority} sample(s); at least 2 samples required."
            )

        # Dynamic neighbor adjustment for small minority cohorts
        effective_k = min(self.smote_k_neighbors, n_minority - 1)
        effective_k = max(1, effective_k)

        smote = SMOTE(
            sampling_strategy=self.sampling_strategy,
            k_neighbors=effective_k,
            random_state=self.random_state,
        )
        tomek = TomekLinks(n_jobs=self.n_jobs)

        resampler = SMOTETomek(
            smote=smote,
            tomek=tomek,
            random_state=self.random_state,
        )

        X_res, y_res = resampler.fit_resample(X, y_arr)

        # Calculate diagnostics
        y_res_arr = np.asarray(y_res)
        unique_res, counts_res = np.unique(y_res_arr, return_counts=True)
        res_dict = dict(zip(unique_res.tolist(), counts_res.tolist()))

        orig_class_0 = int(counts[unique_classes == 0][0]) if 0 in unique_classes else 0
        orig_class_1 = int(counts[unique_classes == 1][0]) if 1 in unique_classes else 0

        res_class_0 = int(res_dict.get(0, 0))
        res_class_1 = int(res_dict.get(1, 0))

        synthetic_generated = max(0, n_majority - n_minority)
        tomek_removed = (orig_class_0 + orig_class_1 + synthetic_generated) - (
            res_class_0 + res_class_1
        )

        self.diagnostics_ = {
            "n_samples_before": int(len(y_arr)),
            "n_samples_after": int(len(y_res_arr)),
            "class_0_before": orig_class_0,
            "class_1_before": orig_class_1,
            "class_0_after": res_class_0,
            "class_1_after": res_class_1,
            "imbalance_ratio_before": float(orig_class_1 / max(orig_class_0, 1)),
            "imbalance_ratio_after": float(res_class_1 / max(res_class_0, 1)),
            "tomek_links_removed": int(max(0, tomek_removed)),
            "effective_k_neighbors": int(effective_k),
        }

        # Preserve original container types
        if is_df:
            X_res = pd.DataFrame(X_res, columns=columns)
        if is_series:
            y_res = pd.Series(y_res, name=target_name)

        return X_res, y_res

    def resample_train_only(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
    ) -> Tuple[
        Union[pd.DataFrame, np.ndarray],
        Union[pd.Series, np.ndarray],
        Union[pd.DataFrame, np.ndarray],
        Union[pd.Series, np.ndarray],
    ]:
        """
        Resample strictly on training partitions while strictly preserving test partitions.

        Parameters
        ----------
        X_train : pd.DataFrame or np.ndarray
            Training features.
        y_train : pd.Series or np.ndarray
            Training labels.
        X_test : pd.DataFrame or np.ndarray
            Test features (MUST NOT be altered).
        y_test : pd.Series or np.ndarray
            Test labels (MUST NOT be altered).

        Returns
        -------
        X_train_res, y_train_res, X_test, y_test
        """
        orig_test_len = len(X_test)
        orig_y_test_len = len(y_test)
        if orig_test_len != orig_y_test_len:
            raise ValueError(
                f"X_test and y_test length mismatch: {orig_test_len} vs {orig_y_test_len}"
            )

        # Copy test distribution snapshot to audit preservation
        orig_y_test_sum = np.sum(np.asarray(y_test))

        # Perform resampling strictly on training data
        X_train_res, y_train_res = self.fit_resample(X_train, y_train)

        # Assert strict test preservation invariant
        assert len(X_test) == orig_test_len, "Test partition size was illegally modified!"
        assert len(y_test) == orig_y_test_len, "Test labels size was illegally modified!"
        assert np.sum(np.asarray(y_test)) == orig_y_test_sum, (
            "Test label distribution was illegally altered!"
        )

        return X_train_res, y_train_res, X_test, y_test

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostic metrics of the latest resampling execution."""
        if self.diagnostics_ is None:
            raise RuntimeError("Resampler has not been fitted yet.")
        return dict(self.diagnostics_)


def resample_training_data(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    random_state: int = 42,
    smote_k_neighbors: int = 5,
    sampling_strategy: Union[float, str] = "auto",
    n_jobs: Optional[int] = None,
) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]:
    """
    Functional convenience wrapper for SMOTE-Tomek balanced resampling.

    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Training feature matrix.
    y_train : pd.Series or np.ndarray
        Training binary labels.
    random_state : int, default=42
        Reproducibility seed.
    smote_k_neighbors : int, default=5
        Nearest neighbors for SMOTE.
    sampling_strategy : Union[float, str], default="auto"
        Sampling strategy for SMOTE.
    n_jobs : Optional[int], default=None
        CPU cores.

    Returns
    -------
    Tuple[X_resampled, y_resampled]
    """
    resampler = SMOTETomekResampler(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
        smote_k_neighbors=smote_k_neighbors,
        n_jobs=n_jobs,
    )
    return resampler.fit_resample(X_train, y_train)
