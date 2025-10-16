"""Mesa-frames implementation of the Boltzmann wealth model with Typer CLI."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import numpy as np
import os
import polars as pl
import typer
from time import perf_counter

from mesa_frames import AgentSet, DataCollector, Model
from examples.utils import FramesSimulationResult
from examples.plotting import plot_model_metrics


# Note: by default we create a timestamped results directory under `results/`.
# The CLI will accept optional `results_dir` and `plots_dir` arguments to override.


def gini(frame: pl.DataFrame) -> float:
    wealth = frame["wealth"] if "wealth" in frame.columns else pl.Series([])
    if wealth.is_empty():
        return float("nan")
    values = wealth.to_numpy().astype(np.float64)
    if values.size == 0:
        return float("nan")
    if np.allclose(values, 0.0):
        return 0.0
    if np.allclose(values, values[0]):
        return 0.0
    sorted_vals = np.sort(values)
    n = sorted_vals.size
    cumulative = np.cumsum(sorted_vals)
    total = cumulative[-1]
    if total == 0:
        return 0.0
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.dot(index, sorted_vals) / (n * total)) - (n + 1) / n)


class MoneyAgents(AgentSet):
    """Vectorised agent set for the Boltzmann wealth exchange model."""

    def __init__(self, model: Model, agents: int) -> None:
        super().__init__(model)
        self += pl.DataFrame({"wealth": pl.Series(np.ones(agents, dtype=np.int64))})

    def step(self) -> None:
        self.select(pl.col("wealth") > 0)
        if len(self.active_agents) == 0:
            return
        # Use the model RNG to seed Polars sampling so results are reproducible
        recipients = self.df.sample(
            n=len(self.active_agents),
            with_replacement=True,
            seed=self.random.integers(np.iinfo(np.int32).max),
        )
        # Combine donor loss (1 per active agent) and recipient gains in a single adjustment.
        gains = recipients.group_by("unique_id").len()
        self.df = (
            self.df.join(gains, on="unique_id", how="left")
            .with_columns(
                (
                    pl.col("wealth")
                    # each active agent loses 1 unit of wealth
                    + pl.when(pl.col("wealth") > 0).then(- 1).otherwise(0)
                    # each agent gains 1 unit of wealth for each time they were selected as a recipient
                    + pl.col("len").fill_null(0)
                ).alias("wealth")
            )
            .drop("len")
        )


class MoneyModel(Model):
    """Mesa-frames model that mirrors the Mesa implementation."""

    def __init__(
        self, agents: int, *, seed: int | None = None, results_dir: Path | None = None
    ) -> None:
        super().__init__(seed)
        self.sets += MoneyAgents(self, agents)
        # For benchmarks we frequently call simulate() without providing a results_dir.
        # Persisting to disk would add unnecessary IO overhead and a missing storage_uri
        # currently raises in DataCollector validation. Fallback to in-memory collection
        # when no results_dir is supplied; otherwise write CSV files under results_dir.
        if results_dir is None:
            storage = "memory"
            storage_uri = None
        else:
            storage = "csv"
            storage_uri = str(results_dir)
        self.datacollector = DataCollector(
            model=self,
            model_reporters={
                "gini": lambda m: gini(m.sets[0].df),
            },
            storage=storage,
            storage_uri=storage_uri,
        )

    def step(self) -> None:
        self.sets.do("step")
        self.datacollector.collect()

    def run(self, steps: int) -> None:
        for _ in range(steps):
            self.step()


def simulate(
    agents: int,
    steps: int,
    seed: int | None = None,
    results_dir: Path | None = None,
) -> FramesSimulationResult:
    model = MoneyModel(agents, seed=seed, results_dir=results_dir)
    model.run(steps)
    # collect data from datacollector into memory first
    return FramesSimulationResult(datacollector=model.datacollector)


app = typer.Typer(add_completion=False)


@app.command()
def run(
    agents: Annotated[int, typer.Option(help="Number of agents to simulate.")] = 5000,
    steps: Annotated[int, typer.Option(help="Number of model steps to run.")] = 100,
    seed: Annotated[int | None, typer.Option(help="Optional RNG seed.")] = None,
    plot: Annotated[bool, typer.Option(help="Render Seaborn plots.")] = True,
    save_results: Annotated[bool, typer.Option(help="Persist metrics as CSV.")] = True,
    results_dir: Annotated[
        Path | None,
        typer.Option(
            help="Directory to write CSV results and plots into. If omitted a timestamped subdir under `results/` is used."
        ),
    ] = None,
) -> None:
    runtime_typechecking = os.environ.get("MESA_FRAMES_RUNTIME_TYPECHECKING", "")
    if runtime_typechecking and runtime_typechecking.lower() not in {"0", "false"}:
        typer.secho(
            "Warning: MESA_FRAMES_RUNTIME_TYPECHECKING is enabled; this run will be slower.",
            fg=typer.colors.YELLOW,
        )
    typer.echo(
        f"Running Boltzmann wealth model (mesa-frames) with {agents} agents for {steps} steps"
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if results_dir is None:
        results_dir = (
            Path(__file__).resolve().parent / "results" / timestamp
        ).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    start_time = perf_counter()
    result = simulate(agents=agents, steps=steps, seed=seed, results_dir=results_dir)

    typer.echo(f"Simulation complete in {perf_counter() - start_time:.2f} seconds")

    model_metrics = result.datacollector.data["model"].select("step", "gini")

    typer.echo(f"Metrics in the final 5 steps: {model_metrics.tail(5)}")

    if save_results:
        result.datacollector.flush()

    if plot:
        stem = f"gini_{timestamp}"
        # write plots into the results directory so outputs are colocated
        plot_model_metrics(
            model_metrics,
            results_dir,
            stem,
            title="Boltzmann wealth — Gini",
            subtitle=f"mesa-frames backend; seed={result.datacollector.seed}",
            agents=agents,
            steps=steps,
        )
        typer.echo(f"Saved plots under {results_dir}")

    # Inform user where CSVs were saved
    typer.echo(f"Saved CSV results under {results_dir}")


if __name__ == "__main__":
    app()
