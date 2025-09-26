"""Mesa implementation of the Boltzmann wealth model with Typer CLI."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Annotated
import pandas as pd

import matplotlib.pyplot as plt
import mesa
from mesa.datacollection import DataCollector
import numpy as np
import polars as pl
import seaborn as sns
import typer
from time import perf_counter

from examples.utils import MesaSimulationResult
from examples.plotting import plot_model_metrics


def gini(values: Iterable[float]) -> float:
    """Compute the Gini coefficient from an iterable of wealth values."""
    array = np.fromiter(values, dtype=float)
    if array.size == 0:
        return float("nan")
    if np.allclose(array, 0.0):
        return 0.0
    if np.allclose(array, array[0]):
        return 0.0
    sorted_vals = np.sort(array)
    n = sorted_vals.size
    cumulative = np.cumsum(sorted_vals)
    total = cumulative[-1]
    if total == 0:
        return 0.0
    index = np.arange(1, n + 1, dtype=float)
    return float((2.0 * np.dot(index, sorted_vals) / (n * total)) - (n + 1) / n)


class MoneyAgent(mesa.Agent):
    """Agent that passes one unit of wealth to a random neighbour."""

    def __init__(self, model: "MoneyModel") -> None:
        super().__init__(model)
        self.wealth = 1

    def step(self) -> None:
        if self.wealth <= 0:
            return
        other = self.random.choice(self.model.agent_list)
        if other is None:
            return
        other.wealth += 1
        self.wealth -= 1


class MoneyModel(mesa.Model):
    """Mesa backend that mirrors the mesa-frames Boltzmann wealth example."""

    def __init__(self, agents: int, *, seed: int | None = None) -> None:
        super().__init__()
        if seed is None:
            seed = self.random.randint(0, np.iinfo(np.int32).max)
        self.reset_randomizer(seed)
        self.agent_list: list[MoneyAgent] = []
        for _ in range(agents):
            # NOTE: storing agents in a Python list keeps iteration fast for benchmarks.
            agent = MoneyAgent(self)
            self.agent_list.append(agent)
        self.datacollector = DataCollector(
            model_reporters={
                "gini": lambda m: gini(a.wealth for a in m.agent_list),
                "seed": lambda m: seed,
            }
        )
        self.datacollector.collect(self)

    def step(self) -> None:
        self.random.shuffle(self.agent_list)
        for agent in self.agent_list:
            agent.step()
        self.datacollector.collect(self)

    def run(self, steps: int) -> None:
        for _ in range(steps):
            self.step()


def simulate(agents: int, steps: int, seed: int | None = None) -> MesaSimulationResult:
    """Run the Mesa Boltzmann wealth model."""
    model = MoneyModel(agents, seed=seed)
    model.run(steps)

    return MesaSimulationResult(datacollector=model.datacollector)


app = typer.Typer(add_completion=False)

@app.command()
def run(
    agents: Annotated[int, typer.Option(help="Number of agents to simulate.")] = 5000,
    steps: Annotated[int, typer.Option(help="Number of model steps to run.")] = 100,
    seed: Annotated[int | None, typer.Option(help="Optional RNG seed.")] = None,
    plot: Annotated[bool, typer.Option(help="Render plots.")] = True,
    save_results: Annotated[
        bool,
        typer.Option(help="Persist metrics as CSV."),
    ] = True,
    results_dir: Annotated[
        Path | None,
        typer.Option(
            help=(
                "Directory to write CSV results and plots into. If omitted a "
                "timestamped subdir under `results/` is used."
            )
        ),
    ] = None,
) -> None:
    """Execute the Mesa Boltzmann wealth simulation."""

    typer.echo(
        f"Running Boltzmann wealth model (mesa) with {agents} agents for {steps} steps"
    )

    # Resolve output folder
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if results_dir is None:
        results_dir = (Path(__file__).resolve().parent / "results" / timestamp).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    start_time = perf_counter()
    # Run simulation (Mesa‑idiomatic): we only use DataCollector's public API
    result = simulate(agents=agents, steps=steps, seed=seed)
    typer.echo(f"Simulation completed in {perf_counter() - start_time:.3f} seconds")
    dc = result.datacollector

    # ---- Extract metrics (no helper, no monkey‑patch):
    # DataCollector returns a pandas DataFrame with the index as the step.
    model_pd = dc.get_model_vars_dataframe()
    model_pd = model_pd.reset_index()
    # The first column is the step index; normalize name to "step".
    model_pd = model_pd.rename(columns={model_pd.columns[0]: "step"})
    seed = model_pd["seed"].iloc[0]
    model_pd = model_pd[['step', 'gini']]

    # Show a short tail in console for quick inspection
    tail_str = model_pd.tail(5).to_string(index=False)
    typer.echo(f"Metrics in the final 5 steps:\n{tail_str}")


    # ---- Save CSV (same filename/layout as frames backend expects)
    if save_results:
        csv_path = results_dir / "model.csv"
        model_pd.to_csv(csv_path, index=False)

    # ---- Plot (convert to Polars to reuse the shared plotting helper)
    if plot and not model_pd.empty:
        model_pl = pl.from_pandas(model_pd)
        stem = f"gini_{timestamp}"
        plot_model_metrics(
            model_pl,
            results_dir,
            stem,
            title="Boltzmann wealth — Gini",
            subtitle=f"mesa backend; seed={seed}",
            agents=agents,
            steps=steps,
        )
        typer.echo(f"Saved plots under {results_dir}")

    if save_results:
        typer.echo(f"Saved CSV results under {results_dir}")


if __name__ == "__main__":
    app()
