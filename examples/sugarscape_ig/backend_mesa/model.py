"""Mesa implementation of Sugarscape IG with Typer CLI (sequential update).

Follows the same structure as the Boltzmann Mesa example: `simulate()` and a
`run` CLI command that saves CSV results and plots the Gini trajectory. The
model updates in the order move -> eat -> regrow -> collect, matching the
tutorial schedule.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Annotated
from time import perf_counter

import mesa
from mesa.datacollection import DataCollector
from mesa.space import SingleGrid
import numpy as np
import pandas as pd
import polars as pl
import typer

from examples.utils import MesaSimulationResult
from examples.plotting import plot_model_metrics

from examples.sugarscape_ig.backend_mesa.agents import AntAgent

def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Safely compute Pearson correlation between two 1-D arrays.

    Mirrors the Frames helper: returns nan for degenerate inputs.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def corr_sugar_metabolism(model: "Sugarscape") -> float:
    sugars = np.fromiter((a.sugar for a in model.agent_list), dtype=float)
    mets = np.fromiter((a.metabolism for a in model.agent_list), dtype=float)
    return _safe_corr(sugars, mets)


def corr_sugar_vision(model: "Sugarscape") -> float:
    sugars = np.fromiter((a.sugar for a in model.agent_list), dtype=float)
    vision = np.fromiter((a.vision for a in model.agent_list), dtype=float)
    return _safe_corr(sugars, vision)


def gini(values: Iterable[float]) -> float:
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


class Sugarscape(mesa.Model):
    def __init__(
        self,
        agents: int,
        *,
        width: int,
        height: int,
        max_sugar: int = 4,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if seed is None:
            seed = self.random.randint(0, np.iinfo(np.int32).max)
        self.reset_randomizer(seed)
        self.width = int(width)
        self.height = int(height)

        # Sugar field (current and max) as 2D arrays shaped (width, height)
        numpy_rng = np.random.default_rng(seed)
        self.sugar_max = numpy_rng.integers(0, max_sugar + 1, size=(width, height), dtype=np.int64)
        self.sugar_current = self.sugar_max.copy()

        # Grid with capacity 1 per cell
        self.grid = SingleGrid(width, height, torus=False)

        # Agents (Python list, manually shuffled/iterated for speed)
        self.agent_list: list[AntAgent] = []
        # Place all agents on empty cells; also draw initial traits from model RNG
        placed = 0
        while placed < agents:
            x = int(self.random.randrange(0, width))
            y = int(self.random.randrange(0, height))
            if self.grid.is_cell_empty((x, y)):
                a = AntAgent(
                    self,
                    sugar=int(self.random.randint(6, 25)),
                    metabolism=int(self.random.randint(2, 5)),
                    vision=int(self.random.randint(1, 6)),
                )
                self.grid.place_agent(a, (x, y))
                self.agent_list.append(a)
                placed += 1

        # Model-level reporters mirroring the Frames implementation so CSVs
        # are comparable across backends.
        self.datacollector = DataCollector(
            model_reporters={
                "mean_sugar": lambda m: float(np.mean([a.sugar for a in m.agent_list])) if m.agent_list else 0.0,
                "total_sugar": lambda m: float(sum(a.sugar for a in m.agent_list)) if m.agent_list else 0.0,
                "agents_alive": lambda m: float(len(m.agent_list)),
                "gini": lambda m: gini(a.sugar for a in m.agent_list),
                "corr_sugar_metabolism": lambda m: corr_sugar_metabolism(m),
                "corr_sugar_vision": lambda m: corr_sugar_vision(m),
                "seed": lambda m: seed,
            },
            agent_reporters={
                "traits": lambda a: {"sugar": a.sugar, "metabolism": a.metabolism, "vision": a.vision}
            },
        )
        self.datacollector.collect(self)

    # --- Scheduling ---

    def _harvest_and_survive(self) -> None:
        survivors: list[AntAgent] = []
        for a in self.agent_list:
            x, y = a.pos
            a.sugar += int(self.sugar_current[x, y])
            a.sugar -= a.metabolism
            # Harvested cells are emptied now; they wil\l be refilled if empty.
            self.sugar_current[x, y] = 0
            if a.sugar > 0:
                survivors.append(a)
            else:
                # Remove dead agent from grid
                self.grid.remove_agent(a)
        self.agent_list = survivors

    def _regrow(self) -> None:
        # Empty cells regrow to max; occupied cells set to 0 (already zeroed on harvest)
        for x in range(self.width):
            for y in range(self.height):
                if self.grid.is_cell_empty((x, y)):
                    self.sugar_current[x, y] = self.sugar_max[x, y]
                else:
                    self.sugar_current[x, y] = 0

    def step(self) -> None:
        # Randomise order, move sequentially, then eat/starve, regrow, collect
        self.random.shuffle(self.agent_list)
        for a in self.agent_list:
            a.move()
        self._harvest_and_survive()
        self._regrow()
        self.datacollector.collect(self)
        if not self.agent_list:
            self.running = False

    def run(self, steps: int) -> None:
        for _ in range(steps):
            if not getattr(self, "running", True):
                break
            self.step()


def simulate(
    *,
    agents: int,
    steps: int,
    width: int,
    height: int,
    max_sugar: int = 4,
    seed: int | None = None,
) -> MesaSimulationResult:
    model = Sugarscape(agents, width=width, height=height, max_sugar=max_sugar, seed=seed)
    model.run(steps)
    return MesaSimulationResult(datacollector=model.datacollector)


app = typer.Typer(add_completion=False)


@app.command()
def run(
    agents: Annotated[int, typer.Option(help="Number of agents to simulate.")] = 400,
    width: Annotated[int, typer.Option(help="Grid width (columns).")] = 40,
    height: Annotated[int, typer.Option(help="Grid height (rows).")] = 40,
    steps: Annotated[int, typer.Option(help="Number of model steps to run.")] = 60,
    max_sugar: Annotated[int, typer.Option(help="Maximum sugar per cell.")] = 4,
    seed: Annotated[int | None, typer.Option(help="Optional RNG seed.")] = None,
    plot: Annotated[bool, typer.Option(help="Render plots.")] = True,
    save_results: Annotated[bool, typer.Option(help="Persist metrics as CSV.")] = True,
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
    typer.echo(
        f"Running Sugarscape IG (mesa, sequential) with {agents} agents on {width}x{height} for {steps} steps"
    )

    # Resolve output folder
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if results_dir is None:
        results_dir = (Path(__file__).resolve().parent / "results" / timestamp).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    start_time = perf_counter()
    result = simulate(agents=agents, steps=steps, width=width, height=height, max_sugar=max_sugar, seed=seed)
    typer.echo(f"Simulation completed in {perf_counter() - start_time:.3f} seconds")
    dc = result.datacollector

    # Extract metrics using DataCollector API
    model_pd = dc.get_model_vars_dataframe().reset_index().rename(columns={"index": "step"})
    # Keep the full model metrics (step + any model reporters)
    seed_val = None
    if "seed" in model_pd.columns and not model_pd.empty:
        seed_val = model_pd["seed"].iloc[0]

    # Show tail for quick inspection (exclude seed column from display)
    display_pd = model_pd.drop(columns=["seed"]) if "seed" in model_pd.columns else model_pd
    typer.echo(f"Metrics in the final 5 steps:\n{display_pd.tail(5).to_string(index=False)}")

    # Save CSV (full model metrics)
    if save_results:
        csv_path = results_dir / "model.csv"
        model_pd.to_csv(csv_path, index=False)

    # Plot per-metric similar to the backend_frames example: create a
    # `plots/` subdirectory and generate one figure per model metric column
    if plot and not model_pd.empty:
        plots_dir = results_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Determine which columns to plot (preserve 'step' if present).
        # Exclude 'seed' from plots so we don't create a chart for a constant
        # model reporter; keep 'seed' in the CSV/dataframe for reproducibility.
        value_cols = [c for c in model_pd.columns if c not in {"step", "seed"}]
        for col in value_cols:
            stem = f"{col}_{timestamp}"
            single = model_pd[["step", col]] if "step" in model_pd.columns else model_pd[[col]]
            # Convert the single-column pandas DataFrame to Polars for the
            # shared plotting helper.
            single_pl = pl.from_pandas(single)
            # Omit seed from subtitle/plot metadata to avoid leaking a constant
            # value into the figure (it remains in the saved CSV). If you want
            # to include the seed in filenames or external metadata, prefer
            # annotating the output folder or README instead.
            plot_model_metrics(
                single_pl,
                plots_dir,
                stem,
                title=f"Sugarscape IG - {col.capitalize()}",
                subtitle="mesa backend",
                agents=agents,
                steps=steps,
            )

        typer.echo(f"Saved plots under {plots_dir}")

    typer.echo(f"Saved CSV results under {results_dir}")


if __name__ == "__main__":
    app()

