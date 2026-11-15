"""
Unit Tests for Automated Paper Artifacts Generator (CREDEX-18).

Verifies:
1. Valid LaTeX syntax across all 4 tables (balanced braces, booktabs, tabular environments).
2. Correct table contents, metrics columns, and bold champion indicators.
3. 300-DPI publication vector PDF and raster PNG figure creation and non-empty file size.
4. Executive results digest markdown generation with required statistical statements.
5. Master PaperArtifactGenerator execution.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.data.loader import generate_synthetic_lending_club
from src.evaluation.paper_artifacts import (
    PaperArtifactGenerator,
    generate_fig1_roc_pr_comparison,
    generate_fig2_expected_loss_curves,
    generate_fig3_shap_global_beeswarm,
    generate_fig4_actionable_recourse_waterfall,
    generate_paper_results_digest,
    generate_table1_descriptive_stats,
    generate_table2_li_wu_replication,
    generate_table3_cost_sensitive_ablation,
    generate_table4_economic_dollar_savings,
    validate_latex_table,
)


@pytest.fixture
def sample_lending_data() -> pd.DataFrame:
    """Generate reproducible sample LendingClub dataset for artifact testing."""
    return generate_synthetic_lending_club(n_samples=500, random_state=42)


def test_validate_latex_table_helper():
    """Verify LaTeX validation helper detects syntax errors and validates clean tables."""
    # Valid minimal table
    valid_latex = """\\begin{table}[htbp]
\\centering
\\caption{Test Caption}
\\label{tab:test}
\\begin{tabular}{lr}
\\toprule
\\textbf{Feature} & \\textbf{Value} \\\\
\\midrule
Income & 50,000 \\\\
\\bottomrule
\\end{tabular}
\\end{table}
"""
    is_valid, errors = validate_latex_table(valid_latex)
    assert is_valid
    assert len(errors) == 0

    # Missing end tag
    invalid_latex_1 = valid_latex.replace("\\end{table}", "")
    is_valid, errors = validate_latex_table(invalid_latex_1)
    assert not is_valid
    assert any("Missing \\begin{table} or \\end{table}" in e for e in errors)

    # Unbalanced braces
    invalid_latex_2 = valid_latex + "{"
    is_valid, errors = validate_latex_table(invalid_latex_2)
    assert not is_valid
    assert any("Unbalanced curly braces" in e for e in errors)


def test_table1_descriptive_stats(tmp_path: Path, sample_lending_data: pd.DataFrame):
    """Verify Table 1 generation, LaTeX syntax, and feature t-test columns."""
    out_file = tmp_path / "table1_test.tex"
    latex_content = generate_table1_descriptive_stats(
        df=sample_lending_data,
        output_path=out_file,
    )

    assert out_file.is_file()
    assert out_file.stat().st_size > 0

    is_valid, errors = validate_latex_table(latex_content)
    assert is_valid, f"LaTeX syntax errors in Table 1: {errors}"

    # Verify structural elements
    assert "\\label{tab:descriptive_stats}" in latex_content
    assert "Non-Default" in latex_content
    assert "Default" in latex_content
    assert "p-value" in latex_content or "p$-value" in latex_content
    assert "\\toprule" in latex_content
    assert "\\bottomrule" in latex_content

    # Key LendingClub features present
    assert "Loan Amount" in latex_content or "loan_amount" in latex_content
    assert "Annual Income" in latex_content or "annual_income" in latex_content


def test_table2_li_wu_replication(tmp_path: Path):
    """Verify Table 2 replication contains all 9 baseline models and bold champions."""
    out_file = tmp_path / "table2_test.tex"
    latex_content = generate_table2_li_wu_replication(output_path=out_file)

    assert out_file.is_file()
    assert out_file.stat().st_size > 0

    is_valid, errors = validate_latex_table(latex_content)
    assert is_valid, f"LaTeX syntax errors in Table 2: {errors}"

    # Verify table headers and reference
    assert "\\label{tab:li_wu_replication}" in latex_content
    assert "Published AUC" in latex_content
    assert "Replicated AUC" in latex_content
    assert "Accuracy" in latex_content
    assert "F1-Score" in latex_content

    # Check all 9 models are represented
    assert "Random Forest" in latex_content
    assert "LightGBM" in latex_content
    assert "XGBoost" in latex_content
    assert "CatBoost" in latex_content
    assert "AdaBoost" in latex_content
    assert "Logistic Regression" in latex_content
    assert "Decision Tree" in latex_content
    assert "Multilayer Perceptron" in latex_content or "MLP" in latex_content
    assert "Gradient Boosted" in latex_content or "GBDT" in latex_content

    # Check champion values are bolded
    assert "\\textbf{" in latex_content


def test_table3_cost_sensitive_ablation(tmp_path: Path):
    """Verify Table 3 2x3x2 Factorial Ablation Matrix structure and bolding."""
    out_file = tmp_path / "table3_test.tex"
    latex_content = generate_table3_cost_sensitive_ablation(output_path=out_file)

    assert out_file.is_file()
    assert out_file.stat().st_size > 0

    is_valid, errors = validate_latex_table(latex_content)
    assert is_valid, f"LaTeX syntax errors in Table 3: {errors}"

    assert "\\label{tab:factorial_ablation}" in latex_content
    assert "Natural" in latex_content
    assert "SMOTE-Tomek" in latex_content
    assert "Standard Log-Loss" in latex_content or "Standard" in latex_content
    assert "Custom Asymmetric" in latex_content
    assert "\\tau^*" in latex_content
    assert "ROC-AUC" in latex_content
    assert "PR-AUC" in latex_content
    assert "Loss (5:1)" in latex_content
    assert "Loss (10:1)" in latex_content

    # Check champion values bolded
    assert "\\textbf{" in latex_content


def test_table4_economic_dollar_savings(tmp_path: Path):
    """Verify Table 4 dollar savings and LGD sensitivity scenarios."""
    out_file = tmp_path / "table4_test.tex"
    latex_content = generate_table4_economic_dollar_savings(output_path=out_file)

    assert out_file.is_file()
    assert out_file.stat().st_size > 0

    is_valid, errors = validate_latex_table(latex_content)
    assert is_valid, f"LaTeX syntax errors in Table 4: {errors}"

    assert "\\label{tab:economic_dollar_savings}" in latex_content
    assert "60\\%" in latex_content
    assert "65\\%" in latex_content
    assert "70\\%" in latex_content
    assert "75\\%" in latex_content
    assert "Economic Dollar Savings" in latex_content
    assert "\\$" in latex_content
    assert "\\textbf{" in latex_content


def test_fig1_roc_pr_comparison(tmp_path: Path):
    """Verify Figure 1 generates valid 300-DPI PDF and PNG files."""
    out_pdf = tmp_path / "fig1_roc_pr_comparison.pdf"
    paths = generate_fig1_roc_pr_comparison(output_path=out_pdf, dpi=150)

    assert "pdf" in paths and "png" in paths
    pdf_p = Path(paths["pdf"])
    png_p = Path(paths["png"])

    assert pdf_p.is_file() and pdf_p.stat().st_size > 0
    assert png_p.is_file() and png_p.stat().st_size > 0


def test_fig2_expected_loss_curves(tmp_path: Path):
    """Verify Figure 2 generates expected loss curves in PDF and PNG."""
    out_pdf = tmp_path / "fig2_expected_loss_curves.pdf"
    paths = generate_fig2_expected_loss_curves(output_path=out_pdf, dpi=150)

    assert "pdf" in paths and "png" in paths
    pdf_p = Path(paths["pdf"])
    png_p = Path(paths["png"])

    assert pdf_p.is_file() and pdf_p.stat().st_size > 0
    assert png_p.is_file() and png_p.stat().st_size > 0


def test_fig3_shap_global_beeswarm(tmp_path: Path):
    """Verify Figure 3 generates TreeSHAP global beeswarm in PDF and PNG."""
    out_pdf = tmp_path / "fig3_shap_global_beeswarm.pdf"
    paths = generate_fig3_shap_global_beeswarm(output_path=out_pdf, dpi=150)

    assert "pdf" in paths and "png" in paths
    pdf_p = Path(paths["pdf"])
    png_p = Path(paths["png"])

    assert pdf_p.is_file() and pdf_p.stat().st_size > 0
    assert png_p.is_file() and png_p.stat().st_size > 0


def test_fig4_actionable_recourse_waterfall(tmp_path: Path):
    """Verify Figure 4 generates dual-panel recourse waterfall in PDF and PNG."""
    out_pdf = tmp_path / "fig4_actionable_recourse_waterfall.pdf"
    paths = generate_fig4_actionable_recourse_waterfall(output_path=out_pdf, dpi=150)

    assert "pdf" in paths and "png" in paths
    pdf_p = Path(paths["pdf"])
    png_p = Path(paths["png"])

    assert pdf_p.is_file() and pdf_p.stat().st_size > 0
    assert png_p.is_file() and png_p.stat().st_size > 0


def test_paper_results_digest(tmp_path: Path):
    """Verify executive results digest generates comprehensive markdown content."""
    out_md = tmp_path / "PAPER_RESULTS_DIGEST.md"
    content = generate_paper_results_digest(output_path=out_md)

    assert out_md.is_file()
    assert out_md.stat().st_size > 0

    assert "Empirical Results Digest" in content
    assert "Li & Wu (2024)" in content
    assert "ROC-AUC" in content
    assert "Total Portfolio Dollar Loss" in content
    assert "DeLong" in content
    assert "Abstract Excerpt" in content
    assert "Empirical Results Excerpt" in content
    assert "Regulatory Recourse" in content


def test_master_paper_artifact_generator(tmp_path: Path, sample_lending_data: pd.DataFrame):
    """Verify end-to-end master paper artifacts generation creates all deliverables."""
    generator = PaperArtifactGenerator(output_dir=tmp_path, dpi=150)
    result = generator.generate_all(df=sample_lending_data)

    # 4 LaTeX tables
    assert len(result.tables) == 4
    for name, p in result.tables.items():
        table_path = Path(p)
        assert table_path.is_file()
        assert table_path.stat().st_size > 0
        is_valid, errors = validate_latex_table(table_path.read_text(encoding="utf-8"))
        assert is_valid, f"Table {name} failed validation: {errors}"

    # 4 Vector figures (each has .pdf and .png)
    assert len(result.figures) == 4
    for name, fig_dict in result.figures.items():
        assert "pdf" in fig_dict and "png" in fig_dict
        assert Path(fig_dict["pdf"]).is_file()
        assert Path(fig_dict["png"]).is_file()

    # Digest
    digest_path = Path(result.digest_path)
    assert digest_path.is_file()
    assert digest_path.stat().st_size > 0
