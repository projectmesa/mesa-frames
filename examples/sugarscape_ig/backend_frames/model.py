"""Mesa-frames implementation of Sugarscape IG with Typer CLI.

This mirrors the advanced tutorial in docs/general/user-guide/3_advanced_tutorial.py
and exposes a simple CLI to run the parallel update variant, save CSVs, and plot
the Gini trajectory.
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Annotated
from time import perf_counter

import numpy as np
import polars as pl
import typer

from mesa_frames import DataCollector, Grid, Model
from examples.utils import FramesSimulationResult
from examples.plotting import plot_model_metrics

from examples.sugarscape_ig.backend_frames.agents import AntsBase, AntsParallel


# Model-level reporters


def gini(model: Model) -> float:
    """Compute the Gini coefficient of agent sugar holdings.

    The function reads the primary agent set from ``model.sets[0]`` and
    computes the population Gini coefficient on the ``sugar`` column. The
    implementation is robust to empty sets and zero-total sugar.

    Parameters
    ----------
    model : Model
        The simulation model that contains agent sets. The primary agent set
        is expected to be at ``model.sets[0]`` and to expose a Polars DataFrame
        under ``.df`` with a ``sugar`` column.

    Returns
    -------
    float
        Gini coefficient in the range [0, 1] if defined, ``0.0`` when the
        total sugar is zero, and ``nan`` when the agent set is empty or too
        small to measure.
    """
    if len(model.sets) == 0:
        return float("nan")

    primary_set = model.sets[0]
    if len(primary_set) == 0:
        return float("nan")

    sugar = primary_set.df["sugar"].to_numpy().astype(np.float64)

    if sugar.size == 0:
        return float("nan")
    sorted_vals = np.sort(sugar.astype(np.float64))
    n = sorted_vals.size
    if n == 0:
        return float("nan")
    cumulative = np.cumsum(sorted_vals)
    total = cumulative[-1]
    if total == 0:
        return 0.0
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.dot(index, sorted_vals) / (n * total)) - (n + 1) / n)


def corr_sugar_metabolism(model: Model) -> float:
    """Pearson correlation between agent sugar and metabolism.

    This reporter extracts the ``sugar`` and ``metabolism`` columns from the
    primary agent set and returns their Pearson correlation coefficient. When
    the agent set is empty or contains insufficient variation the function
    returns ``nan``.

    Parameters
    ----------
    model : Model
        The simulation model that contains agent sets. The primary agent set
        is expected to be at ``model.sets[0]`` and provide a Polars DataFrame
        with ``sugar`` and ``metabolism`` columns.

    Returns
    -------
    float
        Pearson correlation coefficient between sugar and metabolism, or
        ``nan`` when the correlation is undefined (empty set or constant
        values).
    """
    if len(model.sets) == 0:
        return float("nan")

    primary_set = model.sets[0]
    if len(primary_set) == 0:
        return float("nan")

    agent_df = primary_set.df
    sugar = agent_df["sugar"].to_numpy().astype(np.float64)
    metabolism = agent_df["metabolism"].to_numpy().astype(np.float64)
    return _safe_corr(sugar, metabolism)


def corr_sugar_vision(model: Model) -> float:
    """Pearson correlation between agent sugar and vision.

    Extracts the ``sugar`` and ``vision`` columns from the primary agent set
    and returns their Pearson correlation coefficient. If the reporter cannot
    compute a meaningful correlation (for example, when the agent set is
    empty or values are constant) it returns ``nan``.

    Parameters
    ----------
    model : Model
        The simulation model that contains agent sets. The primary agent set
        is expected to be at ``model.sets[0]`` and provide a Polars DataFrame
        with ``sugar`` and ``vision`` columns.

    Returns
    -------
    float
        Pearson correlation coefficient between sugar and vision, or ``nan``
        when the correlation is undefined.
    """
    if len(model.sets) == 0:
        return float("nan")

    primary_set = model.sets[0]
    if len(primary_set) == 0:
        return float("nan")

    agent_df = primary_set.df
    sugar = agent_df["sugar"].to_numpy().astype(np.float64)
    vision = agent_df["vision"].to_numpy().astype(np.float64)
    return _safe_corr(sugar, vision)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Safely compute Pearson correlation between two 1-D arrays.

    This helper guards against degenerate inputs (too few observations or
    constant arrays) which would make the Pearson correlation undefined or
    numerically unstable. When a valid correlation can be computed the
    function returns a Python float.

    Parameters
    ----------
    x : np.ndarray
        One-dimensional numeric array containing the first variable to
        correlate.
    y : np.ndarray
        One-dimensional numeric array containing the second variable to
        correlate.

    Returns
    -------
    float
        Pearson correlation coefficient as a Python float, or ``nan`` if the
        correlation is undefined (fewer than 2 observations or constant
        inputs).
    """
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


class Sugarscape(Model):
    """Minimal Sugarscape model used throughout the tutorial.

    This class wires together a grid that stores ``sugar`` per cell, an
    agent set implementation (passed in as ``agent_type``), and a
    data collector that records model- and agent-level statistics.

    The model's responsibilities are to:
    - create the sugar landscape (cells with current and maximum sugar)
    - create and place agents on the grid
    - advance the sugar regrowth rule each step
    - run the model for a fixed number of steps and collect data

    Parameters
    ----------
    agent_type : type[AntsBase]
        The :class:`AgentSet` subclass implementing the movement rules
        (sequential, numba-accelerated, or parallel).
    n_agents : int
        Number of agents to create and place on the grid.
    width : int
        Grid width (number of columns).
    height : int
        Grid height (number of rows).
    max_sugar : int, optional
        Upper bound for the randomly initialised sugar values on the grid,
        by default 4.
    seed : int | None, optional
        RNG seed to make runs reproducible across variants, by default None.
    results_dir : Path | None, optional
        Optional directory where CSV/plot outputs will be written. If ``None``
        the model runs without persisting CSVs to disk (in-memory storage).

    Notes
    -----
    The grid uses a von Neumann neighbourhood and capacity 1 (at most one
    agent per cell). Both the sugar landscape and initial agent traits are
    drawn from ``self.random`` so different movement variants can be
    instantiated with identical initial conditions by passing the same seed.
    """

    def __init__(
        self,
        agent_type: type[AntsBase],
        n_agents: int,
        *,
        width: int,
        height: int,
        max_sugar: int = 4,
        seed: int | None = None,
        results_dir: Path | None = None,
    ) -> None:
        if n_agents > width * height:
            raise ValueError(
                "Cannot place more agents than grid cells when capacity is 1."
            )
        super().__init__(seed)

        # 1. Let's create the sugar grid and set up the space

        sugar_grid_df = self._generate_sugar_grid(width, height, max_sugar)
        self.space = Grid(
            self, [width, height], neighborhood_type="von_neumann", capacity=1
        )
        self.space.set_cells(sugar_grid_df)
        self._max_sugar = sugar_grid_df.select(["dim_0", "dim_1", "max_sugar"])

        # 2. Now we create the agents and place them on the grid

        agent_frame = self._generate_agent_frame(n_agents)
        main_set = agent_type(self, agent_frame)
        self.sets += main_set
        self.space.place_to_empty(self.sets)

        # 3. Finally we set up the data collector
        # Benchmarks may run without providing a results_dir; in that case avoid forcing
        # a CSV storage backend (which requires a storage_uri) and keep data in memory.
        if results_dir is None:
            storage = "memory"
            storage_uri = None
        else:
            storage = "csv"
            storage_uri = str(results_dir)
        self.datacollector = DataCollector(
            model=self,
            model_reporters={
                "mean_sugar": lambda m: 0.0
                if len(m.sets[0]) == 0
                else float(m.sets[0].df["sugar"].mean()),
                "total_sugar": lambda m: float(m.sets[0].df["sugar"].sum())
                if len(m.sets[0])
                else 0.0,
                "agents_alive": lambda m: float(len(m.sets[0])) if len(m.sets) else 0.0,
                "gini": gini,
                "corr_sugar_metabolism": corr_sugar_metabolism,
                "corr_sugar_vision": corr_sugar_vision,
            },
            agent_reporters={
                "sugar": "sugar",
                "metabolism": "metabolism",
                "vision": "vision",
            },
            storage=storage,
            storage_uri=storage_uri,
        )
        self.datacollector.collect()

    def _generate_sugar_grid(
        self, width: int, height: int, max_sugar: int
    ) -> pl.DataFrame:
        """Generate a random sugar grid.

        Parameters
        ----------
        width : int
            Grid width (number of columns).
        height : int
            Grid height (number of rows).
        max_sugar : int
            Maximum sugar value (inclusive) for each cell.

        Returns
        -------
        pl.DataFrame
            DataFrame with columns ``dim_0``, ``dim_1``, ``sugar`` (current
            amount) and ``max_sugar`` (regrowth target).
        """
        sugar_vals = self.random.integers(
            0, max_sugar + 1, size=(width, height), dtype=np.int64
        )
        dim_0 = pl.Series("dim_0", pl.arange(width, eager=True)).to_frame()
        dim_1 = pl.Series("dim_1", pl.arange(height, eager=True)).to_frame()
        return dim_0.join(dim_1, how="cross").with_columns(
            sugar=sugar_vals.flatten(), max_sugar=sugar_vals.flatten()
        )

    def _generate_agent_frame(self, n_agents: int) -> pl.DataFrame:
        """Create the initial agent frame populated with agent traits.

        Parameters
        ----------
        n_agents : int
            Number of agents to create.

        Returns
        -------
        pl.DataFrame
            DataFrame with columns ``sugar``, ``metabolism`` and ``vision``
            (integer values) for each agent.
        """
        rng = self.random
        return pl.DataFrame(
            {
                "sugar": rng.integers(6, 25, size=n_agents, dtype=np.int64),
                "metabolism": rng.integers(2, 5, size=n_agents, dtype=np.int64),
                "vision": rng.integers(1, 6, size=n_agents, dtype=np.int64),
            }
        )

    def step(self) -> None:
        """Advance the model by one step.

        Notes
        -----
        The per-step ordering is important and this tutorial implements the
        classic Sugarscape "instant growback": agents move and eat first,
        and then empty cells are refilled immediately (move -> eat -> regrow
        -> collect).
        """
        if len(self.sets[0]) == 0:
            self.running = False
            return
        self.sets[0].step()
        self._advance_sugar_field()
        self.datacollector.collect()
        if len(self.sets[0]) == 0:
            self.running = False

    def run(self, steps: int) -> None:
        """Run the model for a fixed number of steps.

        Parameters
        ----------
        steps : int
            Maximum number of steps to run. The model may terminate earlier if
            ``self.running`` is set to ``False`` (for example, when all agents
            have died).
        """
        for _ in range(steps):
            if not self.running:
                break
            self.step()

    def _advance_sugar_field(self) -> None:
        """Apply the instant-growback sugar regrowth rule.

        Empty cells (no agent present) are refilled to their ``max_sugar``
        value. Cells that are occupied are set to zero because agents harvest
        the sugar when they eat. The method uses vectorised DataFrame joins
        and writes to keep the operation efficient.
        """
        empty_cells = self.space.empty_cells
        if not empty_cells.is_empty():
            # Look up the maximum sugar for each empty cell and restore it.
            refresh = empty_cells.join(
                self._max_sugar, on=["dim_0", "dim_1"], how="left"
            )
            self.space.set_cells(empty_cells, {"sugar": refresh["max_sugar"]})
        full_cells = self.space.full_cells
        if not full_cells.is_empty():
            # Occupied cells have just been harvested; set their sugar to 0.
            zeros = pl.Series(np.zeros(len(full_cells), dtype=np.int64))
            self.space.set_cells(full_cells, {"sugar": zeros})


def simulate(
    *,
    agents: int,
    steps: int,
    width: int,
    height: int,
    max_sugar: int = 4,
    seed: int | None = None,
    results_dir: Path | None = None,
) -> FramesSimulationResult:
    model = Sugarscape(
        agent_type=AntsParallel,
        n_agents=agents,
        width=width,
        height=height,
        max_sugar=max_sugar,
        seed=seed,
        results_dir=results_dir,
    )
    model.run(steps)
    return FramesSimulationResult(datacollector=model.datacollector)


app = typer.Typer(add_completion=False)


@app.command()
def run(
    agents: Annotated[int, typer.Option(help="Number of agents to simulate.")] = 400,
    width: Annotated[int, typer.Option(help="Grid width (columns).")] = 40,
    height: Annotated[int, typer.Option(help="Grid height (rows).")] = 40,
    steps: Annotated[int, typer.Option(help="Number of model steps to run.")] = 60,
    max_sugar: Annotated[int, typer.Option(help="Maximum sugar per cell.")] = 4,
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
    typer.echo(
        f"Running Sugarscape IG (mesa-frames, parallel) with {agents} agents on {width}x{height} for {steps} steps"
    )
    runtime_typechecking = os.environ.get("MESA_FRAMES_RUNTIME_TYPECHECKING", "")
    if runtime_typechecking and runtime_typechecking.lower() not in {"0", "false"}:
        typer.secho(
            "Warning: MESA_FRAMES_RUNTIME_TYPECHECKING is enabled; this run will be slower.",
            fg=typer.colors.YELLOW,
        )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if results_dir is None:
        results_dir = (
            Path(__file__).resolve().parent / "results" / timestamp
        ).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    start_time = perf_counter()
    result = simulate(
        agents=agents,
        steps=steps,
        width=width,
        height=height,
        max_sugar=max_sugar,
        seed=seed,
        results_dir=results_dir,
    )
    typer.echo(f"Simulation complete in {perf_counter() - start_time:.2f} seconds")

    model_metrics = result.datacollector.data["model"].drop(["seed", "batch"])
    typer.echo(f"Metrics in the final 5 steps: {model_metrics.tail(5)}")

    if save_results:
        result.datacollector.flush()

    if plot:
        # Create a subdirectory for per-metric plots under the timestamped
        # results directory. For each column in the model metrics (except
        # the step index) create a single-metric DataFrame and call the
        # shared plotting helper to export light/dark PNG+SVG variants.
        plots_dir = results_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Determine which columns to plot (preserve 'step' if present).
        value_cols = [c for c in model_metrics.columns if c != "step"]
        for col in value_cols:
            stem = f"{col}_{timestamp}"
            single = (
                model_metrics.select(["step", col])
                if "step" in model_metrics.columns
                else model_metrics.select([col])
            )
            plot_model_metrics(
                single,
                plots_dir,
                stem,
                title=f"Sugarscape IG — {col.capitalize()}",
                subtitle=f"mesa-frames backend; seed={result.datacollector.seed}",
                agents=agents,
                steps=steps,
            )

        typer.echo(f"Saved plots under {plots_dir}")

    typer.echo(f"Saved CSV results under {results_dir}")


if __name__ == "__main__":
    app()
