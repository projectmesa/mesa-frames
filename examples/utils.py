"""Utilities shared by the examples package.

This module centralises small utilities used across the examples so they
don't have to duplicate simple data containers like SimulationResult.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import polars as pl
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import seaborn as sns



@dataclass
class SimulationResult:
    """Container for example simulation outputs.

    The dataclass is intentionally permissive: some backends only provide
    `metrics`, while others also return `agent_metrics`.
    """

    model_metrics: pl.DataFrame
    agent_metrics: Optional[pl.DataFrame] = None


def plot_model_metrics(
    metrics: pl.DataFrame,
    output_dir: Path,
    stem: str,
    title: str | None = None,
    figsize: tuple[int, int] | None = None,
) -> None:
    """Plot time-series metrics from a polars DataFrame.

    This helper auto-detects all columns except the `step` column and
    plots them as separate series. It writes two theme variants
    (light/dark) as PNG files under ``output_dir`` with the provided stem.
    """
    if metrics.is_empty():
        return

    if "step" not in metrics.columns:
        metrics = metrics.with_row_count("step")

    # melt all non-step columns into long form
    value_cols: Sequence[str] = [c for c in metrics.columns if c != "step"]
    if not value_cols:
        return
    long = metrics.select(["step", *value_cols]).melt(
        id_vars="step", variable_name="metric", value_name="value"
    )

    for theme, style in {"light": "whitegrid", "dark": "darkgrid"}.items():
        sns.set_theme(style=style)
        fig, ax = plt.subplots(figsize=figsize or (8, 5))
        sns.lineplot(data=long.to_pandas(), x="step", y="value", hue="metric", ax=ax)
        ax.set_title(title or "Metrics")
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        fig.tight_layout()
        filename = output_dir / f"{stem}_{theme}.png"
        fig.savefig(filename, dpi=300)
        plt.close(fig)


def plot_agent_metrics(
    agent_metrics: pl.DataFrame, output_dir: Path, stem: str, figsize: tuple[int, int] | None = None
) -> None:
    """Plot agent-level metrics (if any) and write theme variants to disk.

    The function will attempt to preserve common id vars like `step`,
    `seed` and `batch` if present; otherwise it uses the first column as
    the id variable when melting.
    """
    if agent_metrics is None or agent_metrics.is_empty():
        return

    # prefer common id_vars if available
    preferred = ["step", "seed", "batch"]
    id_vars = [c for c in preferred if c in agent_metrics.columns]
    if not id_vars:
        # fall back to using the first column as id
        id_vars = [agent_metrics.columns[0]]

    melted = agent_metrics.melt(id_vars=id_vars, variable_name="metric", value_name="value")

    for theme, style in {"light": "whitegrid", "dark": "darkgrid"}.items():
        sns.set_theme(style=style)
        fig, ax = plt.subplots(figsize=figsize or (10, 6))
        sns.lineplot(data=melted.to_pandas(), x=id_vars[0], y="value", hue="metric", ax=ax)
        ax.set_title("Agent metrics")
        ax.set_xlabel(id_vars[0].capitalize())
        ax.set_ylabel("Value")
        fig.tight_layout()
        filename = output_dir / f"{stem}_agents_{theme}.png"
        fig.savefig(filename, dpi=300)
        plt.close(fig)


__all__ = ["SimulationResult", "plot_model_metrics", "plot_agent_metrics"]
