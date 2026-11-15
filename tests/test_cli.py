"""
Unit Tests for Modular CLI Orchestrator (CREDEX-19).

Verifies:
1. Help strings for all subcommands (ingest, replicate-paper, run-ablation, explain, export-paper, run-all).
2. Proper argument parsing and dispatching.
3. Successful execution of each individual subcommand with small test sample sizes.
4. End-to-end one-shot execution via `run-all` returning exit code 0.
"""

from pathlib import Path

import pytest

from src.cli import main


def test_cli_help_flag(capsys: pytest.CaptureFixture):
    """Verify credex --help prints usage and exits cleanly with 0."""
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "usage: credex" in captured.out
    assert "ingest" in captured.out
    assert "replicate-paper" in captured.out
    assert "run-ablation" in captured.out
    assert "explain" in captured.out
    assert "export-paper" in captured.out
    assert "run-all" in captured.out


@pytest.mark.parametrize(
    "subcommand",
    ["ingest", "replicate-paper", "run-ablation", "explain", "export-paper", "run-all"],
)
def test_cli_subcommand_help(subcommand: str, capsys: pytest.CaptureFixture):
    """Verify each subcommand provides dedicated help text and exits 0."""
    with pytest.raises(SystemExit) as excinfo:
        main([subcommand, "--help"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert f"usage: credex {subcommand}" in captured.out


def test_cli_invalid_command():
    """Verify invalid subcommand raises SystemExit with non-zero error."""
    with pytest.raises(SystemExit) as excinfo:
        main(["invalid_command_xyz"])
    assert excinfo.value.code != 0


def test_cli_ingest_execution():
    """Verify `credex ingest` executes data ingestion and preprocessing stage."""
    # Run with small sample size
    main(["ingest", "--sample-size", "50"])


def test_cli_replicate_paper_execution():
    """Verify `credex replicate-paper` runs baseline replication."""
    main(
        [
            "replicate-paper",
            "--sample-size",
            "50",
            "--n-splits",
            "2",
            "--models",
            "logistic_regression",
        ]
    )


def test_cli_run_ablation_execution(tmp_path: Path):
    """Verify `credex run-ablation` runs factorial ablation matrix."""
    main(
        [
            "run-ablation",
            "--sample-size",
            "50",
            "--n-splits",
            "2",
            "--models",
            "lightgbm",
            "--output-dir",
            str(tmp_path),
        ]
    )


def test_cli_explain_execution():
    """Verify `credex explain` runs TreeSHAP and counterfactual recourse."""
    main(
        [
            "explain",
            "--sample-size",
            "50",
            "--sample-n",
            "20",
        ]
    )


def test_cli_export_paper_execution(tmp_path: Path):
    """Verify `credex export-paper` compiles LaTeX tables, vector figures, and paper digest."""
    main(
        [
            "export-paper",
            "--sample-size",
            "50",
            "--output-dir",
            str(tmp_path),
        ]
    )

    tables_dir = tmp_path / "tables"
    figures_dir = tmp_path / "figures"

    # Verify tables generated
    assert (tables_dir / "table1_descriptive_stats.tex").is_file()
    assert (tables_dir / "table2_li_wu_replication.tex").is_file()
    assert (tables_dir / "table3_cost_sensitive_ablation.tex").is_file()
    assert (tables_dir / "table4_economic_dollar_savings.tex").is_file()

    # Verify figures generated
    assert (figures_dir / "fig1_roc_pr_comparison.pdf").is_file()
    assert (figures_dir / "fig2_expected_loss_curves.pdf").is_file()
    assert (figures_dir / "fig3_shap_global_beeswarm.pdf").is_file()
    assert (figures_dir / "fig4_actionable_recourse_waterfall.pdf").is_file()

    # Verify digest generated
    assert (tmp_path / "PAPER_RESULTS_DIGEST.md").is_file()


def test_cli_run_all_master_execution(tmp_path: Path):
    """Verify `credex run-all` master one-shot execution from end to end."""
    main(
        [
            "run-all",
            "--sample-size",
            "50",
            "--n-splits",
            "2",
            "--models",
            "lightgbm",
            "--sample-n",
            "20",
            "--output-dir",
            str(tmp_path),
        ]
    )

    # Assert end-to-end artifacts exist
    assert (tmp_path / "tables" / "table1_descriptive_stats.tex").is_file()
    assert (tmp_path / "tables" / "table2_li_wu_replication.tex").is_file()
    assert (tmp_path / "tables" / "table3_cost_sensitive_ablation.tex").is_file()
    assert (tmp_path / "tables" / "table4_economic_dollar_savings.tex").is_file()
    assert (tmp_path / "figures" / "fig1_roc_pr_comparison.pdf").is_file()
    assert (tmp_path / "figures" / "fig2_expected_loss_curves.pdf").is_file()
    assert (tmp_path / "figures" / "fig3_shap_global_beeswarm.pdf").is_file()
    assert (tmp_path / "figures" / "fig4_actionable_recourse_waterfall.pdf").is_file()
    assert (tmp_path / "PAPER_RESULTS_DIGEST.md").is_file()
