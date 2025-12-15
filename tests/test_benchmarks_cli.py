from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from benchmarks import cli


def _register_dummy_model(monkeypatch: pytest.MonkeyPatch) -> None:
    # Small, fast no-op backends to avoid running real simulations in tests
    b1 = cli.Backend(name="mesa", runner=lambda agents, steps, seed=None: None)
    b2 = cli.Backend(name="frames", runner=lambda agents, steps, seed=None: None)
    monkeypatch.setitem(cli.MODELS, "dummy", cli.ModelConfig(name="dummy", backends=[b1, b2]))


def _pick_timestamp_dir(base: Path) -> Path:
    # After a run that writes, the base results dir will contain a single timestamp subdir
    subs = [p for p in base.iterdir() if p.is_dir()]
    assert len(subs) <= 1
    return subs[0] if subs else base


def test_summary_save_and_plot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    _register_dummy_model(monkeypatch)
    cli.run(models=["dummy"], agents=[1], steps=1, repeats=1, seed=1, save=True, plot=True, results_dir=tmp_path)
    out = capsys.readouterr().out
    assert "Unified benchmark outputs written:" in out
    assert "CSVs under" in out and "plots under" in out
    ts = _pick_timestamp_dir(tmp_path)
    # CSV should be present
    assert any(ts.glob("*_perf_*.csv"))


def test_summary_plot_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    _register_dummy_model(monkeypatch)
    cli.run(models=["dummy"], agents=[1], steps=1, repeats=1, seed=1, save=False, plot=True, results_dir=tmp_path)
    out = capsys.readouterr().out
    assert "Unified benchmark outputs written:" in out
    assert "plots under" in out and "CSVs under" not in out
    ts = _pick_timestamp_dir(tmp_path)
    # No CSVs should be present, but plots subdir should exist
    assert not any(ts.glob("*_perf_*.csv"))
    assert (ts / "plots").exists()


def test_summary_save_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    _register_dummy_model(monkeypatch)
    cli.run(models=["dummy"], agents=[1], steps=1, repeats=1, seed=1, save=True, plot=False, results_dir=tmp_path)
    out = capsys.readouterr().out
    assert "Unified benchmark outputs written:" in out
    assert "CSVs under" in out and "plots under" not in out
    ts = _pick_timestamp_dir(tmp_path)
    # CSV should be present, plots subdir should not exist
    assert any(ts.glob("*_perf_*.csv"))
    assert not (ts / "plots").exists()


def test_summary_neither(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    _register_dummy_model(monkeypatch)
    cli.run(models=["dummy"], agents=[1], steps=1, repeats=1, seed=1, save=False, plot=False, results_dir=tmp_path)
    out = capsys.readouterr().out
    assert "Benchmark run completed (save=False, plot=False; no files written)." in out
    # No directories created
    assert not any(p for p in tmp_path.iterdir())
