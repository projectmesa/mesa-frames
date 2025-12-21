from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from benchmarks import cli

runner = CliRunner()


def test_benchmarks_cli_runs_minimal(tmp_path: Path) -> None:
    result = runner.invoke(
        cli.app,
        [
            "--models",
            "boltzmann",
            "--agents",
            "10",
            "--steps",
            "1",
            "--repeats",
            "1",
            "--seed",
            "1",
            "--no-save",
            "--no-plot",
            "--results-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "Finished benchmarking model boltzmann" in result.stdout
