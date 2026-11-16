"""
Unit and Integration Tests for Centralized Modular CLI (CREDEX-19).
"""

import pytest
from src.cli import main


def test_cli_global_help(capsys):
    """Verify CLI global help displays usage and subcommands."""
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "credex" in captured.out
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
def test_cli_subcommand_help(subcommand: str, capsys):
    """Verify each CLI subcommand parses --help cleanly."""
    with pytest.raises(SystemExit) as exc:
        main([subcommand, "--help"])
    assert exc.value.code == 0


def test_cli_export_paper_execution(tmp_path):
    """Verify export-paper subcommand runs and creates output directory structure."""
    out_dir = tmp_path / "paper_test_output"
    main(["export-paper", "--sample-size", "200", "--output-dir", str(out_dir)])
    assert out_dir.exists()
    assert (out_dir / "tables").exists()
    assert (out_dir / "figures").exists()
    assert (out_dir / "PAPER_RESULTS_DIGEST.md").exists()
