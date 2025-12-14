"""Typer CLI for running mesa vs mesa-frames performance benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
from time import perf_counter
from typing import Literal, Annotated, Protocol, Optional

import math
import polars as pl
import typer

from examples.boltzmann_wealth import backend_frames as boltzmann_frames
from examples.boltzmann_wealth import backend_mesa as boltzmann_mesa
from examples.sugarscape_ig.backend_frames import model as sugarscape_frames
from examples.sugarscape_ig.backend_mesa import model as sugarscape_mesa
from examples.plotting import (
    plot_performance as _examples_plot_performance,
)

app = typer.Typer(add_completion=False)


class RunnerP(Protocol):
    def __call__(self, agents: int, steps: int, seed: int | None = None) -> None: ...


@dataclass(slots=True)
class Backend:
    name: Literal["mesa", "frames"]
    runner: RunnerP


@dataclass(slots=True)
class ModelConfig:
    name: str
    backends: list[Backend]


MODELS: dict[str, ModelConfig] = {
    "boltzmann": ModelConfig(
        name="boltzmann",
        backends=[
            Backend(name="mesa", runner=boltzmann_mesa.simulate),
            Backend(name="frames", runner=boltzmann_frames.simulate),
        ],
    ),
    "sugarscape": ModelConfig(
        name="sugarscape",
        backends=[
            Backend(
                name="mesa",
                runner=lambda agents, steps, seed=None: sugarscape_mesa.simulate(
                    agents=agents,
                    steps=steps,
                    width=int(max(20, math.ceil((agents) ** 0.5) * 2)),
                    height=int(max(20, math.ceil((agents) ** 0.5) * 2)),
                    seed=seed,
                ),
            ),
            Backend(
                name="frames",
                # Benchmarks expect a runner signature (agents:int, steps:int, seed:int|None)
                # Sugarscape frames simulate requires width/height; choose square close to agent count.
                runner=lambda agents, steps, seed=None: sugarscape_frames.simulate(
                    agents=agents,
                    steps=steps,
                    width=int(max(20, math.ceil((agents) ** 0.5) * 2)),
                    height=int(max(20, math.ceil((agents) ** 0.5) * 2)),
                    seed=seed,
                ),
            ),
        ],
    ),
}


def _parse_agents(value: str) -> list[int]:
    value = value.strip()
    if ":" in value:
        parts = value.split(":")
        if len(parts) != 3:
            raise typer.BadParameter("Ranges must use start:stop:step format")
        try:
            start, stop, step = (int(part) for part in parts)
        except ValueError as exc:
            raise typer.BadParameter("Range values must be integers") from exc
        if step <= 0:
            raise typer.BadParameter("Step must be positive")
        # We keep start = 0 to benchmark initialization time
        if start < 0 or stop <= 0:
            raise typer.BadParameter("Range endpoints must be positive")
        if start > stop:
            raise typer.BadParameter("Range start must be <= stop")
        counts = list(range(start, stop + step, step))
        if counts[-1] > stop:
            counts.pop()
        return counts
    try:
        agents = int(value)
    except ValueError as exc:  # pragma: no cover - defensive
        raise typer.BadParameter("Agent count must be an integer") from exc
    if agents <= 0:
        raise typer.BadParameter("Agent count must be positive")
    return [agents]


def _parse_models(value: str) -> list[str]:
    """Parse models option into a list of model keys.

    Accepts:
    - "all" -> returns all available model keys
    - a single model name -> returns [name]
    - a comma-separated list of model names -> returns list

    Validates that each selected model exists in MODELS.
    """
    value = value.strip()
    if value == "all":
        return list(MODELS.keys())
    # support comma-separated lists
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise typer.BadParameter("Model selection must not be empty")
    unknown = [p for p in parts if p not in MODELS]
    if unknown:
        raise typer.BadParameter(f"Unknown model selection: {', '.join(unknown)}")
    # preserve order and uniqueness
    seen = set()
    result: list[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result


def _plot_performance(
    df: pl.DataFrame, model_name: str, output_dir: Path, timestamp: str
) -> None:
    """Wrap examples.plotting.plot_performance to ensure consistent theming.

    The original benchmark implementation used simple seaborn styles (whitegrid / darkgrid).
    Our example plotting utilities define a much darker, high-contrast *true* dark theme
    (custom rc params overriding bg/fg colors). Reuse that logic here so the
    benchmark dark plots match the example dark plots users see elsewhere.
    """
    if df.is_empty():
        return
    stem = f"{model_name}_runtime_{timestamp}"
    _examples_plot_performance(
        df.select(["agents", "runtime_seconds", "backend"]),
        output_dir=output_dir,
        stem=stem,
        # Prefer more concise, publication-style wording
        title=f"{model_name.title()} runtime scaling",
    )


@app.command()
def run(
    models: Annotated[
        str | list[str],
        typer.Option(
            help="Models to benchmark: boltzmann, sugarscape, or all",
            callback=_parse_models,
        ),
    ] = "all",
    agents: Annotated[
        str | list[int],
        typer.Option(
            help="Agent count or range (start:stop:step)", callback=_parse_agents
        ),
    ] = "1000:5000:1000",
    steps: Annotated[
        int,
        typer.Option(
            min=0,
            help="Number of steps per run.",
        ),
    ] = 100,
    repeats: Annotated[int, typer.Option(help="Repeats per configuration.", min=1)] = 1,
    seed: Annotated[int, typer.Option(help="Optional RNG seed.")] = 42,
    save: Annotated[bool, typer.Option(help="Persist benchmark CSV results.")] = True,
    plot: Annotated[bool, typer.Option(help="Render performance plots.")] = True,
    results_dir: Annotated[
        Optional[Path],
        typer.Option(
            help=(
                "Base directory for benchmark outputs. A timestamped subdirectory "
                "(e.g. results/20250101_120000) is created with CSV files at the root "
                "and a 'plots/' subfolder for images. Defaults to the module's results directory."
            ),
        ),
    ] = None,
) -> None:
    """Run performance benchmarks for the selected models."""
    # Support both CLI (via callbacks) and direct function calls
    if isinstance(models, str):
        models = _parse_models(models)
    if isinstance(agents, str):
        agents = _parse_agents(agents)
    # Ensure module-relative default is computed at call time (avoids import-time side effects)
    if results_dir is None:
        results_dir = Path(__file__).resolve().parent / "results"

    runtime_typechecking = os.environ.get("MESA_FRAMES_RUNTIME_TYPECHECKING", "")
    if runtime_typechecking and runtime_typechecking.lower() not in {"0", "false"}:
        typer.secho(
            "Warning: MESA_FRAMES_RUNTIME_TYPECHECKING is enabled; benchmarks may run significantly slower.",
            fg=typer.colors.YELLOW,
        )
    rows: list[dict[str, object]] = []
    # Single timestamp per CLI invocation so all model results are co-located.
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    # Create unified output layout: <results_dir>/<timestamp>/{CSV files, plots/}
    base_results_dir = results_dir
    timestamp_dir = (base_results_dir / timestamp).resolve()
    plots_subdir: Path = timestamp_dir / "plots"
    for model in models:
        config = MODELS[model]
        typer.echo(f"Benchmarking {model} with agents {agents}")
        for agents_count in agents:
            for repeat_idx in range(repeats):
                run_seed = seed + repeat_idx
                for backend in config.backends:
                    start = perf_counter()
                    backend.runner(agents_count, steps, run_seed)
                    runtime = perf_counter() - start
                    rows.append(
                        {
                            "model": model,
                            "backend": backend.name,
                            "agents": agents_count,
                            "steps": steps,
                            "seed": run_seed,
                            "repeat_idx": repeat_idx,
                            "runtime_seconds": runtime,
                            "timestamp": timestamp,
                        }
                    )
                    # Report completion of this run to the CLI
                    typer.echo(
                        f"Completed {backend.name} for model={model} agents={agents_count} steps={steps} seed={run_seed} repeat={repeat_idx} in {runtime:.3f}s"
                    )
        # Finished all runs for this model
        typer.echo(f"Finished benchmarking model {model}")

    if not rows:
        typer.echo("No benchmark data collected.")
        return
    df = pl.DataFrame(rows)
    if save:
        timestamp_dir.mkdir(parents=True, exist_ok=True)
        for model in models:
            model_df = df.filter(pl.col("model") == model)
            csv_path = timestamp_dir / f"{model}_perf_{timestamp}.csv"
            model_df.write_csv(csv_path)
            typer.echo(f"Saved {model} results to {csv_path}")
    if plot:
        plots_subdir.mkdir(parents=True, exist_ok=True)
        for model in models:
            model_df = df.filter(pl.col("model") == model)
            _plot_performance(model_df, model, plots_subdir, timestamp)
            typer.echo(f"Saved {model} plots under {plots_subdir}")

    typer.echo(
        f"Unified benchmark outputs written under {timestamp_dir} (CSV files) and {plots_subdir} (plots)"
    )


if __name__ == "__main__":
    app()
