"""
Comprehensive Unit Tests and Property-Based Tests for Imbalance Resampling
and Factorial Ablation Matrix (CREDEX-13 & CREDEX-15).

Verifies:
1. SMOTE-Tomek resampling:
   - Balances minority default class while Tomek links eliminate borderline noise.
   - Strictly preserves test partitions without data leakage or corruption.
   - Preserves DataFrame column schemas and Series names.
   - Handles small minority cohorts via adaptive k-neighbors.
2. Factorial Ablation Matrix runner:
   - All 12 cells (2 Sampling x 3 Objective x 2 Cutoff) execute cleanly.
   - Model instantiation across all 4 champion models and 3 objectives.
   - Stratified CV fold aggregation (mean & std).
   - Out-of-Time (OOT) holdout evaluation.
   - Export of structured JSON and CSV run artifacts in `experiments/runs/`.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.data.resampler import (
    SMOTETomekResampler,
    resample_training_data,
)
from src.evaluation.ablation_runner import (
    CHAMPION_MODELS,
    FACTORIAL_CUTOFF_OPTIONS,
    FACTORIAL_OBJECTIVE_OPTIONS,
    FACTORIAL_SAMPLING_OPTIONS,
    FactorialAblationRunner,
    instantiate_ablation_model,
    run_factorial_ablation,
)


# ==============================================================================
# Unit Tests: SMOTE-Tomek Resampler (CREDEX-13)
# ==============================================================================


def test_smote_tomek_balances_classes() -> None:
    """Test that SMOTE-Tomek balances classes and records diagnostics."""
    np.random.seed(42)
    # 100 samples: 85 good loans (0), 15 defaulting loans (1)
    X = np.random.randn(100, 4)
    y = np.array([0] * 85 + [1] * 15)

    resampler = SMOTETomekResampler(random_state=42, smote_k_neighbors=5)
    X_res, y_res = resampler.fit_resample(X, y)

    counts_orig = np.bincount(y)
    counts_res = np.bincount(y_res)

    # Imbalance ratio before was ~0.176; after resampling it should be balanced (~1.0)
    assert counts_res[1] > counts_orig[1]
    assert abs(counts_res[0] - counts_res[1]) <= 5  # Tomek links may remove a few pairs

    diag = resampler.get_diagnostics()
    assert diag["class_0_before"] == 85
    assert diag["class_1_before"] == 15
    assert diag["class_1_after"] > 15
    assert diag["imbalance_ratio_after"] >= 0.85
    assert "tomek_links_removed" in diag


def test_resample_train_only_preserves_test_partition() -> None:
    """
    Test strict leakage prevention invariant:
    Test feature and label partitions MUST remain completely identical.
    """
    np.random.seed(42)
    X_train = pd.DataFrame(np.random.randn(80, 3), columns=["f1", "f2", "f3"])
    y_train = pd.Series([0] * 65 + [1] * 15, name="target")

    X_test = pd.DataFrame(np.random.randn(30, 3), columns=["f1", "f2", "f3"])
    y_test = pd.Series([0] * 24 + [1] * 6, name="target")

    # Snapshot test state
    test_features_snapshot = X_test.copy()
    test_labels_snapshot = y_test.copy()

    resampler = SMOTETomekResampler(random_state=42)
    X_tr_res, y_tr_res, X_te_out, y_te_out = resampler.resample_train_only(
        X_train, y_train, X_test, y_test
    )

    # Training partition is resampled
    assert len(X_tr_res) > len(X_train)
    assert len(y_tr_res) > len(y_train)

    # Test partition is untouched
    assert len(X_te_out) == len(X_test)
    assert len(y_te_out) == len(y_test)
    pd.testing.assert_frame_equal(X_te_out, test_features_snapshot)
    pd.testing.assert_series_equal(y_te_out, test_labels_snapshot)


def test_smote_tomek_preserves_dataframe_types() -> None:
    """Test that DataFrame columns and Series names are preserved."""
    X_df = pd.DataFrame(
        {"annual_income": [50000.0] * 40 + [30000.0] * 10, "dti": [15.0] * 40 + [35.0] * 10}
    )
    y_series = pd.Series([0] * 40 + [1] * 10, name="loan_status")

    X_res, y_res = resample_training_data(X_df, y_series, random_state=42)

    assert isinstance(X_res, pd.DataFrame)
    assert isinstance(y_res, pd.Series)
    assert list(X_res.columns) == ["annual_income", "dti"]
    assert y_res.name == "loan_status"


def test_smote_tomek_small_minority_adaptive_k() -> None:
    """Test adaptive neighbor handling when minority count is very small."""
    X = np.random.randn(30, 3)
    # Only 3 minority instances (less than default k_neighbors=5)
    y = np.array([0] * 27 + [1] * 3)

    resampler = SMOTETomekResampler(random_state=42, smote_k_neighbors=5)
    X_res, y_res = resampler.fit_resample(X, y)

    assert len(X_res) > len(X)
    assert np.sum(y_res == 1) > 3


def test_smote_tomek_invalid_labels_raise() -> None:
    """Test that non-binary target labels raise ValueError."""
    X = np.random.randn(30, 3)
    y_multi = np.array([0] * 10 + [1] * 10 + [2] * 10)

    resampler = SMOTETomekResampler()
    with pytest.raises(ValueError, match="binary target labels"):
        resampler.fit_resample(X, y_multi)


# ==============================================================================
# Unit Tests: Model Instantiation
# ==============================================================================


@pytest.mark.parametrize("model_name", ["lightgbm", "xgboost", "catboost", "random_forest"])
@pytest.mark.parametrize("objective", ["standard", "class_weighted", "asymmetric_custom"])
def test_instantiate_ablation_model_all_combinations(
    model_name: str, objective: str
) -> None:
    """Verify that all 4 champion models instantiate cleanly for all 3 objectives."""
    clf = instantiate_ablation_model(
        model_name=model_name,
        objective=objective,
        w=5.0,
        random_state=42,
        n_estimators=5,
        iterations=5,
    )
    assert clf is not None
    assert hasattr(clf, "fit")
    assert hasattr(clf, "predict_proba")


# ==============================================================================
# Unit Tests: Factorial Ablation Matrix Execution (CREDEX-15)
# ==============================================================================


def test_run_cv_12_cell_matrix_execution(synthetic_lending_club_data: pd.DataFrame) -> None:
    """
    Test execution of the 12-cell Factorial Ablation Matrix across Stratified CV folds.
    Using fast_mode=True and a single champion model (lightgbm) to keep run time sub-minute.
    """
    df = synthetic_lending_club_data.iloc[:100].copy()

    runner = FactorialAblationRunner(
        models=["lightgbm"],
        cost_ratio=5.0,
        random_state=42,
    )

    # 2 folds for rapid unit testing
    results = runner.run_cv(
        df=df,
        target_col="loan_status",
        n_splits=2,
        fast_mode=True,
    )

    assert results["experiment"] == "2x3x2_factorial_ablation_matrix"
    assert results["n_splits"] == 2
    # 2 sampling x 3 objective x 2 cutoff = 12 cells per model
    assert len(results["cells"]) == 12

    # Verify that each cell has valid summary metrics
    for cell in results["cells"]:
        assert cell["model"] == "lightgbm"
        assert cell["sampling"] in FACTORIAL_SAMPLING_OPTIONS
        assert cell["objective"] in FACTORIAL_OBJECTIVE_OPTIONS
        assert cell["cutoff"] in FACTORIAL_CUTOFF_OPTIONS

        summary = cell["summary"]
        assert "roc_auc" in summary
        assert 0.0 <= summary["roc_auc"]["mean"] <= 1.0
        assert 0.0 <= summary["brier_score"]["mean"] <= 1.0
        assert summary["expected_unit_loss_5_1"]["mean"] >= 0.0
        assert len(cell["folds"]) == 2


def test_run_oot_holdout_execution(synthetic_lending_club_data: pd.DataFrame) -> None:
    """Test Out-of-Time holdout evaluation of the factorial matrix."""
    df = synthetic_lending_club_data.copy()

    runner = FactorialAblationRunner(
        models=["random_forest"],
        cost_ratio=5.0,
        random_state=42,
    )

    oot_results = runner.run_oot(
        df=df,
        date_col="issue_d",
        target_col="loan_status",
        train_start="2012-01-01",
        train_end="2016-12-31",
        test_start="2017-01-01",
        test_end="2018-12-31",
        fast_mode=True,
    )

    assert oot_results["type"] == "out_of_time_holdout"
    # 2 sampling x 3 objective x 2 cutoff = 12 cells
    assert len(oot_results["cells"]) == 12

    for cell in oot_results["cells"]:
        metrics = cell["metrics"]
        assert "roc_auc" in metrics
        assert "portfolio_dollar_loss" in metrics
        assert 0.0 <= metrics["brier_score"] <= 1.0


def test_export_results_json_and_csv(
    tmp_path: Path, synthetic_lending_club_data: pd.DataFrame
) -> None:
    """Test that run artifacts (JSON and CSV) are cleanly exported to disk."""
    df = synthetic_lending_club_data.iloc[:80].copy()

    runner = FactorialAblationRunner(
        models=["random_forest"],
        cost_ratio=5.0,
        output_dir=str(tmp_path),
    )

    summary_df, raw_results = runner.run_full_matrix(
        df=df,
        target_col="loan_status",
        n_splits=2,
        evaluate_oot=False,
        fast_mode=True,
        prefix="test_ablation_run",
    )

    # Check DataFrame structure
    assert isinstance(summary_df, pd.DataFrame)
    assert len(summary_df) == 12
    assert "model" in summary_df.columns
    assert "sampling" in summary_df.columns
    assert "objective" in summary_df.columns
    assert "cutoff" in summary_df.columns
    assert "roc_auc_mean" in summary_df.columns
    assert "dollar_savings_pct_mean" in summary_df.columns

    # Check files on disk
    json_file = tmp_path / "test_ablation_run.json"
    csv_file = tmp_path / "test_ablation_run.csv"
    assert json_file.exists()
    assert csv_file.exists()

    with open(json_file, "r") as f:
        data = json.load(f)
        assert "cv_results" in data
        assert len(data["cv_results"]["cells"]) == 12


def test_run_factorial_ablation_convenience(tmp_path: Path) -> None:
    """Test the top-level convenience runner function."""
    summary_df, raw_dict = run_factorial_ablation(
        models=["random_forest"],
        n_splits=2,
        evaluate_oot=False,
        fast_mode=True,
        output_dir=str(tmp_path),
        prefix="convenience_run",
    )

    assert isinstance(summary_df, pd.DataFrame)
    assert len(summary_df) == 12
    assert (tmp_path / "convenience_run.json").exists()
    assert (tmp_path / "convenience_run.csv").exists()
