"""
Unit and Property Tests for Automated Paper Artifacts (CREDEX-18) and CLI (CREDEX-19).
"""

from pathlib import Path
import pytest
from src.evaluation.paper_artifacts import (
    generate_table1_descriptive_stats,
    generate_table2_replication,
    generate_table4_dollar_savings,
    generate_paper_results_digest,
)
from src.cli import main


def test_table1_descriptive_stats_latex_generation(synthetic_lending_club_data, tmp_path):
    """Verify Table 1 LaTeX string generates valid tabular structure."""
    out_file = tmp_path / "table1.tex"
    table_str = generate_table1_descriptive_stats(
        synthetic_lending_club_data,
        feature_cols=["debt_to_income", "annual_income", "loan_amount"],
        target_col="loan_status",
        output_path=out_file,
    )
    assert "\\begin{table}" in table_str
    assert "\\begin{tabular*}" in table_str
    assert "\\caption{Descriptive Statistics" in table_str
    assert "\\end{table}" in table_str
    assert out_file.exists()


def test_table2_replication_latex_generation(tmp_path):
    """Verify Table 2 replication comparison table syntax."""
    out_file = tmp_path / "table2.tex"
    data = {
        "random_forest": {"roc_auc": 0.835, "accuracy": 0.81, "recall": 0.70, "f1": 0.74},
        "lightgbm": {"roc_auc": 0.831, "accuracy": 0.80, "recall": 0.69, "f1": 0.73},
    }
    table_str = generate_table2_replication(data, output_path=out_file)
    assert "\\caption{Empirical Replication of Li \\& Wu (2024)" in table_str
    assert "Random Forest" in table_str
    assert "Lightgbm" in table_str
    assert out_file.exists()


def test_table4_dollar_savings_latex(tmp_path):
    """Verify Table 4 dollar savings table formatting."""
    out_file = tmp_path / "table4.tex"
    data = [
        {"lgd": 0.65, "baseline_dollar_loss": 30000000.0, "enhanced_dollar_loss": 25000000.0, "savings_dollar": 5000000.0, "savings_pct": 16.7}
    ]
    table_str = generate_table4_dollar_savings(data, output_path=out_file)
    assert "65\\%" in table_str
    assert "\\$5,000,000" in table_str
    assert "16.7\\%" in table_str
    assert out_file.exists()


def test_paper_results_digest_generation(tmp_path):
    """Verify markdown results digest compiles."""
    out_file = tmp_path / "PAPER_RESULTS_DIGEST.md"
    stats_dict = {
        "rf_replicated_auc": 0.835,
        "credex_champion_auc": 0.842,
        "dollar_savings_millions": 4.65,
    }
    p = generate_paper_results_digest(stats_dict, output_path=out_file)
    assert p.exists()
    text = p.read_text()
    assert "Credex: Research Paper Empirical Results Digest" in text
    assert "$4.65 Million" in text


def test_cli_ingest_help(capsys):
    """Verify CLI help parses correctly."""
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
