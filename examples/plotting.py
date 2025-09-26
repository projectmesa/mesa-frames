# examples/plotting.py
from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence
import re

import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# ----------------------------- Shared theme ----------------------------------

_THEMES = {
    "light": dict(
        style="whitegrid",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
        },
    ),
    "dark": dict(
        style="whitegrid",
        rc={
            # real dark background + readable foreground
            "figure.facecolor": "#0b1021",
            "axes.facecolor": "#0b1021",
            "axes.edgecolor": "#d6d6d7",
            "axes.labelcolor": "#e8e8ea",
            "text.color": "#e8e8ea",
            "xtick.color": "#c9c9cb",
            "ytick.color": "#c9c9cb",
            "grid.color": "#2a2f4a",
            "grid.alpha": 0.35,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.facecolor": "#121734",
            "legend.edgecolor": "#3b3f5a",
        },
    ),
}


def _shorten_seed(text: str | None) -> str | None:
    """Turn '... seed=1234567890123' into '... seed=12345678…' if present."""
    if not text:
        return text
    m = re.search(r"seed=([^;,\s]+)", text)
    if not m:
        return text
    raw = m.group(1)
    short = (raw[:8] + "…") if len(raw) > 10 else raw
    return re.sub(r"seed=[^;,\s]+", f"seed={short}", text)


def _apply_titles(fig: Figure, ax: Axes, title: str, subtitle: str | None) -> None:
    """Consistent title placement: figure-level title + small italic subtitle."""
    fig.suptitle(title, fontsize=18, y=0.98)
    ax.set_title(_shorten_seed(subtitle) or "", fontsize=12, fontstyle="italic", pad=4)


def _finalize_and_save(fig: Figure, output_dir: Path, stem: str, theme: str) -> None:
    """Tight layout with space for suptitle, export PNG + (optional) SVG."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    png = output_dir / f"{stem}_{theme}.png"
    fig.savefig(png, dpi=300)
    try:
        fig.savefig(output_dir / f"{stem}_{theme}.svg", bbox_inches="tight")
    except Exception:
        pass  # SVG is a nice-to-have
    plt.close(fig)


# -------------------------- Public: model metrics ----------------------------


def plot_model_metrics(
    metrics: pl.DataFrame,
    output_dir: Path,
    stem: str,
    title: str,
    *,
    subtitle: str = "",
    figsize: tuple[int, int] | None = None,
    agents: int | None = None,
    steps: int | None = None,
) -> None:
    """
    Plot time-series metrics from a Polars DataFrame and export light/dark PNG/SVG.

    - Auto-detects `step` or adds one if missing.
    - Melts all non-`step` columns into long form.
        - If there's a single metric (e.g., 'gini'), removes legend and uses a
            descriptive y-axis label (e.g., 'Gini coefficient').
        - Optional `agents` and `steps` will be appended to the suptitle as
            "(N=<agents>, T=<steps>)"; if `steps` is omitted it will be inferred
            from the `step` column when available.
    """
    if metrics.is_empty():
        return

    if "step" not in metrics.columns:
        metrics = metrics.with_row_index("step")

    # If steps not provided, try to infer from the data (max step + 1). Keep it None if we can't determine it.
    if steps is None:
        try:
            steps = int(metrics.select(pl.col("step").max()).item()) + 1
        except Exception:
            steps = None

    value_cols: Sequence[str] = [c for c in metrics.columns if c != "step"]
    if not value_cols:
        return

    long = (
        metrics.select(["step", *value_cols])
        .unpivot(
            index="step", on=value_cols, variable_name="metric", value_name="value"
        )
        .to_pandas()
    )

    # Compose informative title with optional (N, T)
    if agents is not None and steps is not None:
        full_title = f"{title} (N={agents}, T={steps})"
    elif agents is not None:
        full_title = f"{title} (N={agents})"
    elif steps is not None:
        full_title = f"{title} (T={steps})"
    else:
        full_title = title

    for theme, cfg in _THEMES.items():
        sns.set_theme(**cfg)
        sns.set_context("talk")
        fig, ax = plt.subplots(figsize=figsize or (10, 6))

        sns.lineplot(data=long, x="step", y="value", hue="metric", linewidth=2, ax=ax)

        _apply_titles(fig, ax, full_title, subtitle)

        ax.set_xlabel("Step")
        unique_metrics = long["metric"].unique()

        if len(unique_metrics) == 1:
            name = unique_metrics[0]
            ax.set_ylabel(name.capitalize())
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
            vals = long.loc[long["metric"] == name, "value"]
            if not vals.empty:
                vmin, vmax = float(vals.min()), float(vals.max())
                pad = max(0.005, (vmax - vmin) * 0.05)
                ax.set_ylim(vmin - pad, vmax + pad)
        else:
            ax.set_ylabel("Value")
            leg = ax.get_legend()
            if theme == "dark" and leg is not None:
                leg.set_title(None)
                leg.get_frame().set_alpha(0.8)

        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.margins(x=0.01)

        _finalize_and_save(fig, output_dir, stem, theme)


# -------------------------- Public: agent metrics ----------------------------


def plot_agent_metrics(
    agent_metrics: pl.DataFrame,
    output_dir: Path,
    stem: str,
    *,
    title: str = "Agent metrics",
    subtitle: str = "",
    figsize: tuple[int, int] | None = None,
) -> None:
    """
    Plot agent-level metrics (multi-series) and export light/dark PNG/SVG.

    - Preserves common id vars if present: `step`, `seed`, `batch`.
    - Uses the first column as id if none of the preferred ids exist.
    """
    if agent_metrics is None or agent_metrics.is_empty():
        return

    preferred = ["step", "seed", "batch"]
    id_vars = [c for c in preferred if c in agent_metrics.columns] or [
        agent_metrics.columns[0]
    ]

    # Determine which columns to unpivot (all columns except the id vars).
    value_cols = [c for c in agent_metrics.columns if c not in id_vars]
    if not value_cols:
        return

    melted = agent_metrics.unpivot(
        index=id_vars, on=value_cols, variable_name="metric", value_name="value"
    ).to_pandas()

    xcol = id_vars[0]

    for theme, cfg in _THEMES.items():
        sns.set_theme(**cfg)
        sns.set_context("talk")
        fig, ax = plt.subplots(figsize=figsize or (10, 6))

        sns.lineplot(data=melted, x=xcol, y="value", hue="metric", linewidth=1.8, ax=ax)

        _apply_titles(fig, ax, title, subtitle)
        ax.set_xlabel(xcol.capitalize())
        ax.set_ylabel("Value")

        if theme == "dark":
            leg = ax.get_legend()
            if leg is not None:
                leg.set_title(None)
                leg.get_frame().set_alpha(0.8)

        _finalize_and_save(fig, output_dir, f"{stem}_agents", theme)


# -------------------------- Public: performance ------------------------------


def plot_performance(
    df: pl.DataFrame,
    output_dir: Path,
    stem: str,
    *,
    title: str = "Runtime vs agents",
    subtitle: str = "",
    figsize: tuple[int, int] | None = None,
) -> None:
    """
    Plot backend performance (runtime vs agents) with mean±sd error bars.
    Expected columns: `agents`, `runtime_seconds`, `backend`.
    """
    if df.is_empty():
        return

    pdf = df.to_pandas()

    for theme, cfg in _THEMES.items():
        sns.set_theme(**cfg)
        sns.set_context("talk")
        fig, ax = plt.subplots(figsize=figsize or (10, 6))

        sns.lineplot(
            data=pdf,
            x="agents",
            y="runtime_seconds",
            hue="backend",
            estimator="mean",
            errorbar="sd",
            marker="o",
            ax=ax,
        )

        _apply_titles(fig, ax, title, subtitle)
        ax.set_xlabel("Agents")
        ax.set_ylabel("Runtime (seconds)")

        if theme == "dark":
            leg = ax.get_legend()
            if leg is not None:
                leg.set_title(None)
                leg.get_frame().set_alpha(0.8)

        _finalize_and_save(fig, output_dir, stem, theme)


__all__ = [
    "plot_model_metrics",
    "plot_agent_metrics",
    "plot_performance",
]
