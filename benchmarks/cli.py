"""Typer CLI for running mesa vs mesa-frames performance benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Literal, Annotated, Protocol, Optional

import math
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import typer

from examples.boltzmann_wealth import backend_frames as boltzmann_frames
from examples.boltzmann_wealth import backend_mesa as boltzmann_mesa
from examples.sugarscape_ig.backend_frames import model as sugarscape_frames
from examples.sugarscape_ig.backend_mesa import model as sugarscape_mesa

app = typer.Typer(add_completion=False)

class RunnerP(Protocol):
    def __call__(self, agents: int, steps: int, seed: Optional[int] = None) -> None: ...


@dataclass(slots=True)
class Backend:
    name: Literal['mesa', 'frames']
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
        if start <= 0 or stop <= 0:
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
    if df.is_empty():
        return
    for theme, style in {"light": "whitegrid", "dark": "darkgrid"}.items():
        sns.set_theme(style=style)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.lineplot(
            data=df.to_pandas(),
            x="agents",
            y="runtime_seconds",
            hue="backend",
            estimator="mean",
            errorbar="sd",
            marker="o",
            ax=ax,
        )
        ax.set_title(f"{model_name.title()} runtime vs agents")
        ax.set_xlabel("Agents")
        ax.set_ylabel("Runtime (seconds)")
        fig.tight_layout()
        filename = output_dir / f"{model_name}_runtime_{timestamp}_{theme}.png"
        fig.savefig(filename, dpi=300)
        plt.close(fig)


@app.command()
def run(
    models: Annotated[str, typer.Option(
        help="Models to benchmark: boltzmann, sugarscape, or all",
        callback=_parse_models
    )] = "all",
    agents: Annotated[str, typer.Option(
        help="Agent count or range (start:stop:step)",
        callback=_parse_agents
    )] = "1000:5000:1000",
    steps: Annotated[int, typer.Option(
        min=0,
        help="Number of steps per run.",
    )] = 100,
    repeats: Annotated[int, typer.Option(help="Repeats per configuration.", min=1)] = 1,
    seed: Annotated[int, typer.Option(help="Optional RNG seed.")] = 42,
    save: Annotated[bool, typer.Option(help="Persist benchmark CSV results.")] = True,
    plot: Annotated[bool, typer.Option(help="Render performance plots.")] = True,
    results_dir: Annotated[Path, typer.Option(
        help="Directory for benchmark CSV results.",
    )] = Path(__file__).resolve().parent / "results",
    plots_dir: Annotated[Path, typer.Option(
        help="Directory for benchmark plots.",
    )] = Path(__file__).resolve().parent / "plots",
) -> None:
    """Run performance benchmarks for the models models."""
    rows: list[dict[str, object]] = []
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
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
    if not rows:
        typer.echo("No benchmark data collected.")
        return
    df = pl.DataFrame(rows)
    if save:
        results_dir.mkdir(parents=True, exist_ok=True)
        for model in models:
            model_df = df.filter(pl.col("model") == model)
            csv_path = results_dir / f"{model}_perf_{timestamp}.csv"
            model_df.write_csv(csv_path)
            typer.echo(f"Saved {model} results to {csv_path}")
    if plot:
        plots_dir.mkdir(parents=True, exist_ok=True)
        for model in models:
            model_df = df.filter(pl.col("model") == model)
            _plot_performance(model_df, model, plots_dir, timestamp)
            typer.echo(f"Saved {model} plots under {plots_dir}")


if __name__ == "__main__":
    app()
