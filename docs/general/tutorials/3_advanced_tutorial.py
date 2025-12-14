from __future__ import annotations

# %% [markdown]
"""
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/projectmesa/mesa-frames/blob/main/docs/general/user-guide/3_advanced_tutorial.ipynb)

# Advanced Tutorial — Rebuilding Sugarscape with mesa-frames

We revisit the classic Sugarscape instant-growback model described in chapter 2 of [Growing Artificial Societies](https://direct.mit.edu/books/monograph/2503/Growing-Artificial-SocietiesSocial-Science-from) (Epstein & Axtell,
1996) and rebuild it step by step using `mesa-frames`. Along the way we highlight why the traditional definition is not ideal for high-performance with mesa-frames and how a simple relaxation can unlock vectorisation and lead to similar macro behaviour.

## Sugarscape in Plain Terms

We model a population of *ants* living on a rectangular grid rich in sugar. Each
cell can host at most one ant and holds a fixed amount of sugar. Every time step
unfolds as follows:

* **Sense:** each ant looks outward along the four cardinal directions up to its
  `vision` radius and spots open cells.
* **Move:** the ant chooses the cell with highest sugar (breaking ties by
  distance and coordinates). In the instant-growback variant used here, any cell
  that was occupied at the end of the previous step has sugar 0 (it was harvested
  and did not regrow).
* **Eat & survive:** ants harvest the sugar on the cell they occupy. If their
  sugar stock falls below their `metabolism` cost, they die.
* **Regrow:** sugar instantly regrows to its maximum level on empty cells. The
  landscape is drawn from a uniform distribution, so resources are homogeneous
  on average and the interesting dynamics come from agent heterogeneity and
  congestion.

The update schedule matters for micro-behaviour, so we study three variants:

1. **Sequential loop (asynchronous):** This is the traditional definition. Ants move one at a time in random order.
This cannot be vectorised easily as the best move for an ant might depend on the moves of earlier ants (for example, if they target the same cell).
2. **Sequential with Numba:** matches the first variant but relies on a compiled
   helper for speed.
3. **Parallel (synchronous):** ants rank candidate destinations using the
   start-of-step sugar field; conflicts are resolved by a random lottery in
   rounds (losers fall back to their next choice). If an ant wins a destination
   other than its origin, its origin becomes available to other ants in later
   rounds of the same step.

The first variant (pure Python loops) is a natural starting point, but it is **not** the mesa-frames philosophy.
The latter two are: we aim to **write rules declaratively** and let the dataframe engine worry about performance.
Our guiding principle is to **focus on modelling first and performance second**. Only when a rule is truly
inherently sequential do we fall back to a compiled kernel (Numba or JAX).

Our goal is to compare these update rules and show how far a synchronous,
dataframe-friendly rule can go as a **performance-oriented relaxation** of the
classic sequential schedule. Some macroscopic summaries (like total sugar or the
Gini coefficient of wealth) often remain qualitatively similar, while more
fine-grained statistics (like wealth–trait correlations) can drift noticeably for
individual seeds because conflict resolution changes which traits win contested
cells.
"""

# %% [markdown]
# First, let's install and import the necessary packages.

# %% [markdown]
# If you're running this tutorial on Google Colab or another fresh environment,
# uncomment the cell below to install the required dependencies.

# %%
# #!pip install git+https://github.com/projectmesa/mesa-frames polars numba numpy

# %% [markdown]
"""## 1. Imports"""

# %%
from time import perf_counter

import numpy as np
import polars as pl
from numba import njit

from mesa_frames import AgentSet, DataCollector, Grid, Model


# Simple display helper to render HTML in notebooks while keeping stdout
# output for scripts/CI runs.
def _in_ipython() -> bool:
    """Return True when running inside an IPython/Jupyter session."""
    try:
        from IPython import get_ipython
    except Exception:
        return False
    return get_ipython() is not None


def show_output(
    obj: object,
    *,
    title: str | None = None,
    max_rows: int = 12,
    collapsible: bool = False,
    open_by_default: bool = False,
) -> None:
    """Display rich HTML when available, otherwise fall back to prints."""
    rich_env = _in_ipython()

    if isinstance(obj, pl.DataFrame):
        df = obj.head(max_rows) if max_rows else obj
        if rich_env:
            from IPython.display import HTML, display

            if collapsible:
                open_attr = " open" if open_by_default else ""
                summary = title or "Table"
                html = df.to_pandas().to_html(index=False)
                display(
                    HTML(
                        f"""
<details{open_attr} style="margin: 0.75em 0;">
  <summary style="cursor:pointer; font-weight:600;">{summary}</summary>
  <div style="margin-top:0.5em;">{html}</div>
</details>
"""
                    )
                )
            else:
                if title:
                    display(HTML(f"<h4 style='margin: 0.6em 0 0.2em'>{title}</h4>"))
                display(df.to_pandas())
        else:
            if title:
                print(f"\n=== {title} ===")
            print(df)
        return

    if title:
        if rich_env:
            from IPython.display import HTML, display

            display(HTML(f"<h4 style='margin: 0.6em 0 0.2em'>{title}</h4>"))
        else:
            print(f"\n=== {title} ===")

    if rich_env:
        from IPython.display import display

        display(obj)
    else:
        print(obj)


# %% [markdown]
"""## 2. Model definition

In this section we define some helpers and the model class that wires
together the grid and the agents. The `agent_type` parameter stays flexible so
we can plug in different movement policies later, but the model now owns the
logic that generates the sugar field and the initial agent frame. Because both
helpers use `self.random`, instantiating each variant with the same seed keeps
the initial conditions identical across the sequential, Numba, and parallel
implementations.

The space is a von Neumann grid (which means agents can only move up, down, left,
or right) with capacity 1, meaning each cell can host at most one agent. The sugar
field is stored as part of the cell data frame, with columns for current sugar
and maximum sugar (for regrowth). The model also sets up a data collector to
track aggregate statistics and agent traits over time.

The `step` method advances the sugar field, triggers the agent set's step.

We also define some useful functions to compute metrics like the Gini coefficient and correlations.
"""


# %%

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
        value. Occupied cells have already been harvested in
        :meth:`AntsBase.eat`, so we only need to refresh empty cells here.
        The method uses vectorised DataFrame joins and writes to keep the
        operation efficient.
        """
        empty_cells = self.space.empty_cells
        if not empty_cells.is_empty():
            # Look up the maximum sugar for each empty cell and restore it.
            refresh = empty_cells.join(
                self._max_sugar, on=["dim_0", "dim_1"], how="left"
            )
            self.space.set_cells(empty_cells, {"sugar": refresh["max_sugar"]})


# %% [markdown]

"""
## 3. Agent definition

### 3.1 Base agent class

Now let's define the agent class (the ant class). We start with a base class which implements the common logic for eating and starvation, while leaving the `move` method abstract.
The base class also provides helper methods for sensing visible cells and choosing the best cell based on sugar, distance, and coordinates.
This will allow us to define different movement policies (sequential, Numba-accelerated, and parallel) as subclasses that only need to implement the `move` method.
"""

# %%


class AntsBase(AgentSet):
    """Base agent set for the Sugarscape tutorial.

    This class implements the common behaviour shared by all agent
    movement variants (sequential, numba-accelerated and parallel).

    Notes
    -----
    - Agents are expected to have integer traits: ``sugar``, ``metabolism``
      and ``vision``. These are validated in :meth:`__init__`.
    - Subclasses must implement :meth:`move` which changes agent positions
      on the grid (via :meth:`mesa_frames.Grid` helpers).
    """

    def __init__(self, model: Model, agent_frame: pl.DataFrame) -> None:
        """Initialise the agent set and validate required trait columns.

        Parameters
        ----------
        model : Model
            The parent model which provides RNG and space.
        agent_frame : pl.DataFrame
            A Polars DataFrame with at least the columns ``sugar``,
            ``metabolism`` and ``vision`` for each agent.

        Raises
        ------
        ValueError
            If required trait columns are missing from ``agent_frame``.
        """
        super().__init__(model)
        required = {"sugar", "metabolism", "vision"}
        missing = required.difference(agent_frame.columns)
        if missing:
            raise ValueError(
                f"Initial agent frame must include columns {sorted(required)}; missing {sorted(missing)}."
            )
        self.add(agent_frame.clone())

    def step(self) -> None:
        """Advance the agent set by one time step.

        The update order is important: agents are first shuffled to randomise
        move order (this is important only for sequential variants), then they move, harvest sugar
        from their occupied cells, and finally any agents whose sugar falls
        to zero or below are removed.
        """
        # Randomise ordering for movement decisions when required by the
        # implementation (e.g. sequential update uses this shuffle).
        self.shuffle(inplace=True)
        # Movement policy implemented by subclasses.
        self.move()
        # Agents harvest sugar on their occupied cells.
        self.eat()
        # Remove agents that starved after eating.
        self._remove_starved()

    def move(self) -> None:  # pragma: no cover
        """Abstract movement method.

        Subclasses must override this method to update agent positions on the
        grid. Implementations should use :meth:`mesa_frames.Grid.move_agents`
        or similar helpers provided by the space API.
        """
        raise NotImplementedError

    def eat(self) -> None:
        """Agents harvest sugar from the cells they currently occupy.

        Behaviour:
        - Look up the set of occupied cells (cells that reference an agent
          id).
        - For each occupied cell, add the cell sugar to the agent's sugar
          stock and subtract the agent's metabolism cost.
        - After agents harvest, set the sugar on those cells to zero (they
          were consumed).
        """
        # Map of currently occupied agent ids on the grid.
        occupied_ids = self.index
        # `occupied_ids` is a Polars Series; calling `is_in` with a Series
        # of the same datatype is ambiguous in newer Polars. Use `implode`
        # to collapse the Series into a list-like value for membership checks.
        occupied_cells = self.space.cells.filter(
            pl.col("agent_id").is_in(occupied_ids.implode())
        )
        if occupied_cells.is_empty():
            return
        # The agent ordering here uses the agent_id values stored in the
        # occupied cells frame; indexing the agent set with that vector updates
        # the matching agents' sugar values in one vectorised write.
        agent_ids = occupied_cells["agent_id"]
        self[agent_ids, "sugar"] = (
            self[agent_ids, "sugar"]
            + occupied_cells["sugar"]
            - self[agent_ids, "metabolism"]
        )
        # After harvesting, occupied cells have zero sugar.
        self.space.set_cells(
            occupied_cells.select(["dim_0", "dim_1"]),
            {"sugar": pl.Series(np.zeros(len(occupied_cells), dtype=np.int64))},
        )

    def _remove_starved(self) -> None:
        """Discard agents whose sugar stock has fallen to zero or below.

        This method performs a vectorised filter on the agent frame and
        removes any matching rows from the set.
        """
        starved = self.df.filter(pl.col("sugar") <= 0)
        if not starved.is_empty():
            # ``discard`` accepts a DataFrame of agents to remove.
            self.discard(starved)


# %% [markdown]

"""### 3.2 Sequential movement

We now implement the simplest movement policy: sequential (asynchronous). Each agent moves one at a time in the current ordering, choosing the best visible cell according to the rules.

This implementation uses plain Python loops as the logic cannot be easily vectorised. As a result, it is slow for large populations and grids. We will later show how to speed it up with Numba.
"""

# %%


class AntsSequential(AntsBase):
    def _visible_cells(
        self, origin: tuple[int, int], vision: int
    ) -> list[tuple[int, int]]:
        """List cells visible from an origin along the four cardinal axes.

        The visibility set includes the origin cell itself and cells at
        Manhattan distances 1..vision along the four cardinal directions
        (up, down, left, right), clipped to the grid bounds.

        Parameters
        ----------
        origin : tuple[int, int]
            The agent's current coordinate ``(x, y)``.
        vision : int
            Maximum Manhattan radius to consider along each axis.

        Returns
        -------
        list[tuple[int, int]]
            Ordered list of visible cells (origin first, then increasing
            step distance along each axis).
        """
        x0, y0 = origin
        width, height = self.space.dimensions
        cells: list[tuple[int, int]] = [origin]
        # Look outward one step at a time in the four cardinal directions.
        for step in range(1, vision + 1):
            if x0 + step < width:
                cells.append((x0 + step, y0))
            if x0 - step >= 0:
                cells.append((x0 - step, y0))
            if y0 + step < height:
                cells.append((x0, y0 + step))
            if y0 - step >= 0:
                cells.append((x0, y0 - step))
        return cells

    def _choose_best_cell(
        self,
        origin: tuple[int, int],
        vision: int,
        sugar_map: dict[tuple[int, int], int],
        blocked: set[tuple[int, int]] | None,
    ) -> tuple[int, int]:
        """Select the best visible cell according to the movement rules.

        Tie-break rules (in order):
        1. Prefer cells with strictly greater sugar.
        2. If equal sugar, prefer the cell with smaller distance from the
           origin (measured with the Frobenius norm returned by
           ``space.get_distances``).
        3. If still tied, prefer the cell with smaller coordinates (lexicographic
           ordering of the ``(x, y)`` tuple).

        Parameters
        ----------
        origin : tuple[int, int]
            Agent's current coordinate.
        vision : int
            Maximum vision radius along cardinal axes.
        sugar_map : dict[tuple[int, int], int]
            Mapping from ``(x, y)`` to sugar amount.
        blocked : set[tuple[int, int]] | None
            Optional set of coordinates that should be considered occupied and
            therefore skipped (except the origin which is always allowed).

        Returns
        -------
        tuple[int, int]
            Chosen target coordinate (may be the origin if no better cell is
            available).
        """
        best_cell = origin
        best_sugar = sugar_map.get(origin, 0)
        best_distance = 0
        ox, oy = origin
        for candidate in self._visible_cells(origin, vision):
            # Skip blocked cells (occupied by other agents) unless it's the
            # agent's current cell which we always consider.
            if blocked and candidate != origin and candidate in blocked:
                continue
            sugar_here = sugar_map.get(candidate, 0)
            # Use step-based Manhattan distance (number of steps along cardinal
            # axes) which is the same metric used by the Numba path. This avoids
            # calling the heavier `space.get_distances` per candidate.
            cx, cy = candidate
            distance = abs(cx - ox) + abs(cy - oy)
            better = False
            # Primary criterion: strictly more sugar.
            if sugar_here > best_sugar:
                better = True
            elif sugar_here == best_sugar:
                # Secondary: closer distance.
                if distance < best_distance:
                    better = True
                # Tertiary: lexicographic tie-break on coordinates.
                elif distance == best_distance and candidate < best_cell:
                    better = True
            if better:
                best_cell = candidate
                best_sugar = sugar_here
                best_distance = distance
        return best_cell

    def _current_sugar_map(self) -> dict[tuple[int, int], int]:
        """Return a mapping from grid coordinates to the current sugar value.

        Returns
        -------
        dict[tuple[int, int], int]
            Keys are ``(x, y)`` tuples and values are the integer sugar amount
            on that cell (zero if missing/None).
        """
        cells = self.space.cells.select(["dim_0", "dim_1", "sugar"])
        # Build a plain Python dict for fast lookups in the movement code.
        return {
            (int(x), int(y)): 0 if sugar is None else int(sugar)
            for x, y, sugar in cells.iter_rows()
        }

    def move(self) -> None:
        sugar_map = self._current_sugar_map()
        state = self.df.join(self.pos, on="unique_id", how="left")
        positions = {
            int(row["unique_id"]): (int(row["dim_0"]), int(row["dim_1"]))
            for row in state.iter_rows(named=True)
        }
        taken: set[tuple[int, int]] = set(positions.values())

        for row in state.iter_rows(named=True):
            agent_id = int(row["unique_id"])
            vision = int(row["vision"])
            current = positions[agent_id]
            taken.discard(current)
            target = self._choose_best_cell(current, vision, sugar_map, taken)
            taken.add(target)
            positions[agent_id] = target
            if target != current:
                self.space.move_agents(agent_id, target)


# %% [markdown]
"""
### 3.3 Speeding Up the Loop with Numba

As we will see later, the previous sequential implementation is slow for large populations and grids because it relies on plain Python loops. We can speed it up significantly by using Numba to compile the movement logic.

Numba compiles numerical Python code to fast machine code at runtime. To use Numba, we need to rewrite the movement logic in a way that is compatible with Numba's restrictions (using tightly typed numpy arrays and accessing data indexes directly).
"""


# %%
@njit(cache=True)
def _numba_should_replace(
    best_sugar: int,
    best_distance: int,
    best_x: int,
    best_y: int,
    candidate_sugar: int,
    candidate_distance: int,
    candidate_x: int,
    candidate_y: int,
) -> bool:
    """Numba helper: decide whether a candidate cell should replace the
    current best cell according to the movement tie-break rules.

    This implements the same ordering used in :meth:`_choose_best_cell` but
    in a tightly-typed, compiled form suitable for Numba loops.

    Parameters
    ----------
    best_sugar : int
        Sugar at the current best cell.
    best_distance : int
        Manhattan distance from the origin to the current best cell.
    best_x : int
        X coordinate of the current best cell.
    best_y : int
        Y coordinate of the current best cell.
    candidate_sugar : int
        Sugar at the candidate cell.
    candidate_distance : int
        Manhattan distance from the origin to the candidate cell.
    candidate_x : int
        X coordinate of the candidate cell.
    candidate_y : int
        Y coordinate of the candidate cell.

    Returns
    -------
    bool
        True if the candidate should replace the current best cell.
    """
    # Primary criterion: prefer strictly greater sugar.
    if candidate_sugar > best_sugar:
        return True
    # If sugar ties, prefer the closer cell.
    if candidate_sugar == best_sugar:
        if candidate_distance < best_distance:
            return True
        # If distance ties as well, compare coordinates lexicographically.
        if candidate_distance == best_distance:
            if candidate_x < best_x:
                return True
            if candidate_x == best_x and candidate_y < best_y:
                return True
    return False


@njit(cache=True)
def _numba_find_best_cell(
    x0: int,
    y0: int,
    vision: int,
    sugar_array: np.ndarray,
    occupied: np.ndarray,
) -> tuple[int, int]:
    width, height = sugar_array.shape
    best_x = x0
    best_y = y0
    best_sugar = sugar_array[x0, y0]
    best_distance = 0

    # Examine visible cells along the four cardinal directions, increasing
    # step by step. The 'occupied' array marks cells that are currently
    # unavailable (True = occupied). The origin cell is allowed as the
    # default; callers typically clear the origin before searching.
    for step in range(1, vision + 1):
        nx = x0 + step
        if nx < width and not occupied[nx, y0]:
            sugar_here = sugar_array[nx, y0]
            if _numba_should_replace(
                best_sugar, best_distance, best_x, best_y, sugar_here, step, nx, y0
            ):
                best_x = nx
                best_y = y0
                best_sugar = sugar_here
                best_distance = step

        nx = x0 - step
        if nx >= 0 and not occupied[nx, y0]:
            sugar_here = sugar_array[nx, y0]
            if _numba_should_replace(
                best_sugar, best_distance, best_x, best_y, sugar_here, step, nx, y0
            ):
                best_x = nx
                best_y = y0
                best_sugar = sugar_here
                best_distance = step

        ny = y0 + step
        if ny < height and not occupied[x0, ny]:
            sugar_here = sugar_array[x0, ny]
            if _numba_should_replace(
                best_sugar, best_distance, best_x, best_y, sugar_here, step, x0, ny
            ):
                best_x = x0
                best_y = ny
                best_sugar = sugar_here
                best_distance = step

        ny = y0 - step
        if ny >= 0 and not occupied[x0, ny]:
            sugar_here = sugar_array[x0, ny]
            if _numba_should_replace(
                best_sugar, best_distance, best_x, best_y, sugar_here, step, x0, ny
            ):
                best_x = x0
                best_y = ny
                best_sugar = sugar_here
                best_distance = step

    return best_x, best_y


@njit(cache=True)
def sequential_move_numba(
    dim0: np.ndarray,
    dim1: np.ndarray,
    vision: np.ndarray,
    sugar_array: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Numba-accelerated sequential movement helper.

    This function emulates the traditional asynchronous (sequential) update
    where agents move one at a time in the current ordering. It accepts
    numpy arrays describing agent positions and vision ranges, and a 2D
    sugar array for lookup.

    Parameters
    ----------
    dim0 : np.ndarray
        1D integer array of length n_agents containing the x coordinates
        for each agent.
    dim1 : np.ndarray
        1D integer array of length n_agents containing the y coordinates
        for each agent.
    vision : np.ndarray
        1D integer array of vision radii for each agent.
    sugar_array : np.ndarray
        2D array shaped (width, height) containing per-cell sugar values.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Updated arrays of x and y coordinates after sequential movement.
    """
    n_agents = dim0.shape[0]
    width, height = sugar_array.shape
    # Copy inputs to avoid mutating caller arrays in-place.
    new_dim0 = dim0.copy()
    new_dim1 = dim1.copy()
    # Occupancy grid: True when a cell is currently occupied by an agent.
    occupied = np.zeros((width, height), dtype=np.bool_)

    # Mark initial occupancy.
    for i in range(n_agents):
        occupied[new_dim0[i], new_dim1[i]] = True

    # Process agents in order. For each agent we clear its current cell in
    # the occupancy grid (so it can consider moving into it), search for the
    # best unoccupied visible cell, and mark the chosen destination as
    # occupied. This models agents moving one-by-one.
    for i in range(n_agents):
        x0 = new_dim0[i]
        y0 = new_dim1[i]
        # Free the agent's current cell so it is considered available during
        # the search (agents may choose to stay, in which case we'll re-mark
        # it below).
        occupied[x0, y0] = False
        best_x, best_y = _numba_find_best_cell(
            x0, y0, int(vision[i]), sugar_array, occupied
        )
        # Claim the chosen destination.
        occupied[best_x, best_y] = True
        new_dim0[i] = best_x
        new_dim1[i] = best_y

    return new_dim0, new_dim1


class AntsNumba(AntsBase):
    def move(self) -> None:
        state = self.df.join(self.pos, on="unique_id", how="left")
        if state.is_empty():
            return
        agent_ids = state["unique_id"]
        dim0 = state["dim_0"].to_numpy().astype(np.int64)
        dim1 = state["dim_1"].to_numpy().astype(np.int64)
        vision = state["vision"].to_numpy().astype(np.int64)

        sugar_array = (
            self.space.cells.sort(["dim_0", "dim_1"])
            .with_columns(pl.col("sugar").fill_null(0))["sugar"]
            .to_numpy()
            .reshape(self.space.dimensions)
        )

        new_dim0, new_dim1 = sequential_move_numba(dim0, dim1, vision, sugar_array)
        coords = pl.DataFrame({"dim_0": new_dim0.tolist(), "dim_1": new_dim1.tolist()})
        self.space.move_agents(agent_ids, coords)


# %% [markdown]
"""
### 3.4 Simultaneous Movement with Conflict Resolution (the Polars mesa-frames idiomatic way)

The previous implementation is optimal speed-wise but it's a bit low-level. It requires maintaining an occupancy grid and imperative loops and it might become tricky to extend with more complex movement rules or models.
To stay in mesa-frames idiom, we can implement a parallel movement policy that uses Polars DataFrame operations to resolve conflicts when multiple agents target the same cell.
These conflicts are resolved in rounds: in each round, each agent proposes its current best candidate cell; winners per cell are chosen at random, and losers are promoted to their next-ranked choice. This continues until all agents have moved.
This implementation is a tad slower but still efficient and easier to read (for a Polars user).
"""

# %%


class AntsParallel(AntsBase):
    def move(self) -> None:
        """Move agents in parallel by ranking visible cells and resolving conflicts.

        Declarative mental model: express *what* each agent wants (ranked candidates),
        then use dataframe ops to *allocate* (joins, group_by with a lottery).
        Performance is handled by Polars/LazyFrames; avoid premature micro-optimisations.

        Returns
        -------
        None
            Movement updates happen in-place on the underlying space.
        """
        # Early exit if there are no agents.
        if len(self.df) == 0:
            return

        # current_pos columns:
        # ┌──────────┬────────────────┬────────────────┐
        # │ agent_id ┆ dim_0_center   ┆ dim_1_center   │
        # │ ---      ┆ ---            ┆ ---            │
        # │ u64      ┆ i64            ┆ i64            │
        # ╞══════════╪════════════════╪════════════════╡
        current_pos = self.pos.select(
            [
                pl.col("unique_id").alias("agent_id"),
                pl.col("dim_0").alias("dim_0_center"),
                pl.col("dim_1").alias("dim_1_center"),
            ]
        )

        neighborhood = self._build_neighborhood_frame(current_pos)
        choices, origins, max_rank = self._rank_candidates(neighborhood, current_pos)
        if choices.is_empty():
            return

        assigned = self._resolve_conflicts_in_rounds(choices, origins, max_rank)
        if assigned.is_empty():
            return

        # move_df columns:
        # ┌────────────┬────────────┬────────────┐
        # │ unique_id  ┆ dim_0      ┆ dim_1      │
        # │ ---        ┆ ---        ┆ ---        │
        # │ u64        ┆ i64        ┆ i64        │
        # ╞════════════╪════════════╪════════════╡
        move_df = pl.DataFrame(
            {
                "unique_id": assigned["agent_id"],
                "dim_0": assigned["dim_0_candidate"],
                "dim_1": assigned["dim_1_candidate"],
            }
        )
        # `move_agents` accepts IdsLike and SpaceCoordinates (Polars Series/DataFrame),
        # so pass Series/DataFrame directly rather than converting to Python lists.
        self.space.move_agents(move_df["unique_id"], move_df.select(["dim_0", "dim_1"]))

    def _build_neighborhood_frame(self, current_pos: pl.DataFrame) -> pl.DataFrame:
        """Assemble the sugar-weighted neighbourhood for each sensing agent.

        Parameters
        ----------
        current_pos : pl.DataFrame
            DataFrame with columns ``agent_id``, ``dim_0_center`` and
            ``dim_1_center`` describing the current position of each agent.

        Returns
        -------
        pl.DataFrame
            DataFrame with columns ``agent_id``, ``radius``, ``dim_0_candidate``,
            ``dim_1_candidate`` and ``sugar`` describing the visible cells for
            each agent.
        """
        # Build a neighbourhood frame: for each agent and visible cell we
        # attach the cell sugar. The raw offsets contain the candidate
        # cell coordinates and the center coordinates for the sensing agent.
        # Raw neighborhood columns:
        # ┌────────────┬────────────┬────────┬────────────────┬────────────────┐
        # │ dim_0      ┆ dim_1      ┆ radius ┆ dim_0_center   ┆ dim_1_center   │
        # │ ---        ┆ ---        ┆ ---    ┆ ---            ┆ ---            │
        # │ i64        ┆ i64        ┆ i64    ┆ i64            ┆ i64            │
        # ╞════════════╪════════════╪════════╪════════════════╪════════════════╡
        neighborhood_cells = self.space.get_neighborhood(
            radius=self["vision"], agents=self, include_center=True
        )

        # sugar_cells columns:
        # ┌────────────┬────────────┬────────┐
        # │ dim_0      ┆ dim_1      ┆ sugar  │
        # │ ---        ┆ ---        ┆ ---    │
        # │ i64        ┆ i64        ┆ i64    │
        # ╞════════════╪════════════╪════════╡

        sugar_cells = self.space.cells.select(["dim_0", "dim_1", "sugar"])

        neighborhood_cells = (
            neighborhood_cells.join(sugar_cells, on=["dim_0", "dim_1"], how="left")
            .with_columns(pl.col("sugar").fill_null(0))
            .rename({"dim_0": "dim_0_candidate", "dim_1": "dim_1_candidate"})
        )

        neighborhood_cells = neighborhood_cells.join(
            current_pos,
            left_on=["dim_0_center", "dim_1_center"],
            right_on=["dim_0_center", "dim_1_center"],
            how="left",
        )

        # Final neighborhood columns:
        # ┌──────────┬────────┬──────────────────┬──────────────────┬────────┐
        # │ agent_id ┆ radius ┆ dim_0_candidate  ┆ dim_1_candidate  ┆ sugar  │
        # │ ---      ┆ ---    ┆ ---              ┆ ---              ┆ ---    │
        # │ u64      ┆ i64    ┆ i64              ┆ i64              ┆ i64    │
        # ╞══════════╪════════╪══════════════════╪══════════════════╪════════╡
        neighborhood_cells = neighborhood_cells.drop(
            ["dim_0_center", "dim_1_center"]
        ).select(["agent_id", "radius", "dim_0_candidate", "dim_1_candidate", "sugar"])

        return neighborhood_cells

    def _rank_candidates(
        self,
        neighborhood: pl.DataFrame,
        current_pos: pl.DataFrame,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Rank candidate destination cells for each agent.

        Parameters
        ----------
        neighborhood : pl.DataFrame
            Output of :meth:`_build_neighborhood_frame` with columns
            ``agent_id``, ``radius``, ``dim_0_candidate``, ``dim_1_candidate``
            and ``sugar``.
        current_pos : pl.DataFrame
            Frame with columns ``agent_id``, ``dim_0_center`` and
            ``dim_1_center`` describing where each agent currently stands.

        Returns
        -------
        choices : pl.DataFrame
            Ranked candidates per agent with columns ``agent_id``,
            ``dim_0_candidate``, ``dim_1_candidate``, ``sugar``, ``radius`` and
            ``rank``.
        origins : pl.DataFrame
            Original coordinates per agent with columns ``agent_id``,
            ``dim_0`` and ``dim_1``.
        max_rank : pl.DataFrame
            Maximum available rank per agent with columns ``agent_id`` and
            ``max_rank``.
        """
        # Create ranked choices per agent: sort by sugar (desc), radius
        # (asc), then coordinates. Keep the first unique entry per cell.

        # choices columns (after select):
        # ┌──────────┬──────────────────┬──────────────────┬────────┬────────┐
        # │ agent_id ┆ dim_0_candidate  ┆ dim_1_candidate  ┆ sugar  ┆ radius │
        # │ ---      ┆ ---              ┆ ---              ┆ ---    ┆ ---    │
        # │ u64      ┆ i64              ┆ i64              ┆ i64    ┆ i64    │
        # ╞══════════╪══════════════════╪══════════════════╪════════╪════════╡
        choices = (
            neighborhood.select(
                [
                    "agent_id",
                    "dim_0_candidate",
                    "dim_1_candidate",
                    "sugar",
                    "radius",
                ]
            )
            .sort(
                ["agent_id", "sugar", "radius", "dim_0_candidate", "dim_1_candidate"],
                descending=[False, True, False, False, False],
            )
            .unique(
                subset=["agent_id", "dim_0_candidate", "dim_1_candidate"],
                keep="first",
                maintain_order=True,
            )
            .with_columns(pl.col("agent_id").cum_count().over("agent_id").alias("rank"))
        )

        # Precompute per-agent candidate rank once so conflict resolution can
        # promote losers by incrementing a cheap `current_rank` counter,
        # without re-sorting after each round. Alternative: drop taken cells
        # and re-rank by sugar every round; simpler conceptually but requires
        # repeated sorts and deduplication, which is heavier than filtering by
        # `rank >= current_rank`.

        # Origins for fallback (if an agent exhausts candidates it stays put).
        # origins columns:
        # ┌──────────┬────────────┬────────────┐
        # │ agent_id ┆ dim_0      ┆ dim_1      │
        # │ ---      ┆ ---        ┆ ---        │
        # │ u64      ┆ i64        ┆ i64        │
        # ╞══════════╪════════════╪════════════╡
        origins = current_pos.select(
            [
                "agent_id",
                pl.col("dim_0_center").alias("dim_0"),
                pl.col("dim_1_center").alias("dim_1"),
            ]
        )

        # Track the maximum available rank per agent to clamp promotions.
        # This bounds `current_rank`; once an agent reaches `max_rank` and
        # cannot secure a cell, they fall back to origin cleanly instead of
        # chasing nonexistent ranks.
        # max_rank columns:
        # ┌──────────┬───────────┐
        # │ agent_id ┆ max_rank │
        # │ ---      ┆ ---       │
        # │ u64      ┆ u32       │
        # ╞══════════╪═══════════╡
        max_rank = choices.group_by("agent_id").agg(
            pl.col("rank").max().alias("max_rank")
        )
        return choices, origins, max_rank

    def _resolve_conflicts_in_rounds(
        self,
        choices: pl.DataFrame,
        origins: pl.DataFrame,
        max_rank: pl.DataFrame,
    ) -> pl.DataFrame:
        """Resolve movement conflicts through iterative lottery rounds.

        Parameters
        ----------
        choices : pl.DataFrame
            Ranked candidate cells per agent with headers matching the
            ``choices`` frame returned by :meth:`_rank_candidates`.
        origins : pl.DataFrame
            Agent origin coordinates with columns ``agent_id``, ``dim_0`` and
            ``dim_1``.
        max_rank : pl.DataFrame
            Maximum rank offset per agent with columns ``agent_id`` and
            ``max_rank``.

        Returns
        -------
        pl.DataFrame
            Allocated movements with columns ``agent_id``, ``dim_0_candidate``
            and ``dim_1_candidate``; each row records the destination assigned
            to an agent.
        """
        # Prepare unresolved agents and working tables.
        agent_ids = choices["agent_id"].unique(maintain_order=True)

        # unresolved columns:
        # ┌──────────┬────────────────┐
        # │ agent_id ┆ current_rank  │
        # │ ---      ┆ ---            │
        # │ u64      ┆ i64            │
        # ╞══════════╪════════════════╡
        unresolved = pl.DataFrame(
            {
                "agent_id": agent_ids,
                "current_rank": pl.Series(np.zeros(len(agent_ids), dtype=np.int64)),
            }
        )

        # assigned columns:
        # ┌──────────┬──────────────────┬──────────────────┐
        # │ agent_id ┆ dim_0_candidate  ┆ dim_1_candidate  │
        # │ ---      ┆ ---              ┆ ---              │
        # │ u64      ┆ i64              ┆ i64              │
        # ╞══════════╪══════════════════╪══════════════════╡
        assigned = pl.DataFrame(
            {
                "agent_id": pl.Series(
                    name="agent_id", values=[], dtype=agent_ids.dtype
                ),
                "dim_0_candidate": pl.Series(
                    name="dim_0_candidate", values=[], dtype=pl.Int64
                ),
                "dim_1_candidate": pl.Series(
                    name="dim_1_candidate", values=[], dtype=pl.Int64
                ),
            }
        )

        # taken columns:
        # ┌──────────────────┬──────────────────┐
        # │ dim_0_candidate  ┆ dim_1_candidate  │
        # │ ---              ┆ ---              │
        # │ i64              ┆ i64              │
        # ╞══════════════════╪══════════════════╡
        # Treat all currently occupied cells (origins) as taken from the start.
        # Each agent may still target its own origin; we handle that exception
        # when filtering candidate pools.
        taken = origins.select(
            [
                pl.col("dim_0").alias("dim_0_candidate"),
                pl.col("dim_1").alias("dim_1_candidate"),
            ]
        )
        origins_for_filter = origins.rename(
            {"dim_0": "dim_0_origin", "dim_1": "dim_1_origin"}
        )

        # Resolve in rounds: each unresolved agent proposes its current-ranked
        # candidate; winners per-cell are selected at random and losers are
        # promoted to their next choice.
        while unresolved.height > 0:
            # Using precomputed `rank` lets us select candidates with
            # `rank >= current_rank` and avoid re-ranking after each round.
            # Alternative: remove taken cells and re-sort remaining candidates
            # by sugar/distance per round (heavier due to repeated sort/dedupe).
            # candidate_pool columns (after join with unresolved):
            # ┌──────────┬──────────────────┬──────────────────┬────────┬────────┬──────┬──────────────┐
            # │ agent_id ┆ dim_0_candidate  ┆ dim_1_candidate  ┆ sugar  ┆ radius ┆ rank ┆ current_rank │
            # │ ---      ┆ ---              ┆ ---              ┆ ---    ┆ ---    ┆ ---  ┆ ---          │
            # │ u64      ┆ i64              ┆ i64              ┆ i64    ┆ i64    ┆ u32  ┆ i64          │
            # ╞══════════╪══════════════════╪══════════════════╪════════╪════════╪══════╪══════════════╡
            candidate_pool = choices.join(unresolved, on="agent_id")
            candidate_pool = candidate_pool.filter(
                pl.col("rank") >= pl.col("current_rank")
            )
            candidate_pool = (
                candidate_pool.join(origins_for_filter, on="agent_id", how="left")
                .join(
                    taken.with_columns(pl.lit(True).alias("is_taken")),
                    on=["dim_0_candidate", "dim_1_candidate"],
                    how="left",
                )
                .filter(
                    pl.col("is_taken").is_null()
                    | (
                        (pl.col("dim_0_candidate") == pl.col("dim_0_origin"))
                        & (pl.col("dim_1_candidate") == pl.col("dim_1_origin"))
                    )
                )
                .drop(["dim_0_origin", "dim_1_origin", "is_taken"])
            )

            if candidate_pool.is_empty():
                # No available candidates — everyone falls back to origin.
                # Note: this covers both agents with no visible cells left and
                # the case where all remaining candidates are already taken.
                # fallback columns:
                # ┌──────────┬────────────┬────────────┬──────────────┐
                # │ agent_id ┆ dim_0      ┆ dim_1      ┆ current_rank │
                # │ ---      ┆ ---        ┆ ---        ┆ ---          │
                # │ u64      ┆ i64        ┆ i64        ┆ i64          │
                # ╞══════════╪════════════╪════════════╪══════════════╡
                fallback = unresolved.join(origins, on="agent_id", how="left")
                assigned = pl.concat(
                    [
                        assigned,
                        fallback.select(
                            [
                                "agent_id",
                                pl.col("dim_0").alias("dim_0_candidate"),
                                pl.col("dim_1").alias("dim_1_candidate"),
                            ]
                        ),
                    ],
                    how="vertical",
                )
                break

            # best_candidates columns (per agent first choice):
            # ┌──────────┬──────────────────┬──────────────────┬────────┬────────┬──────┬──────────────┐
            # │ agent_id ┆ dim_0_candidate  ┆ dim_1_candidate  ┆ sugar  ┆ radius ┆ rank ┆ current_rank │
            # │ ---      ┆ ---              ┆ ---              ┆ ---    ┆ ---    ┆ ---  ┆ ---          │
            # │ u64      ┆ i64              ┆ i64              ┆ i64    ┆ i64    ┆ u32  ┆ i64          │
            # ╞══════════╪══════════════════╪══════════════════╪════════╪════════╪══════╪══════════════╡
            best_candidates = (
                candidate_pool.sort(["agent_id", "rank"])
                .group_by("agent_id", maintain_order=True)
                .first()
            )

            # Agents that had no candidate this round fall back to origin.
            # missing columns:
            # ┌──────────┬──────────────┐
            # │ agent_id ┆ current_rank │
            # │ ---      ┆ ---          │
            # │ u64      ┆ i64          │
            # ╞══════════╪══════════════╡
            missing = unresolved.join(
                best_candidates.select("agent_id"), on="agent_id", how="anti"
            )
            if not missing.is_empty():
                # fallback (missing) columns match fallback table above.
                fallback = missing.join(origins, on="agent_id", how="left")
                assigned = pl.concat(
                    [
                        assigned,
                        fallback.select(
                            [
                                "agent_id",
                                pl.col("dim_0").alias("dim_0_candidate"),
                                pl.col("dim_1").alias("dim_1_candidate"),
                            ]
                        ),
                    ],
                    how="vertical",
                )
                unresolved = unresolved.join(
                    missing.select("agent_id"), on="agent_id", how="anti"
                )
                best_candidates = best_candidates.join(
                    missing.select("agent_id"), on="agent_id", how="anti"
                )
                if unresolved.is_empty() or best_candidates.is_empty():
                    continue

            # Add a small random lottery to break ties deterministically for
            # each candidate set.
            lottery = pl.Series("lottery", self.random.random(best_candidates.height))
            best_candidates = best_candidates.with_columns(lottery)

            # winners columns:
            # ┌──────────┬──────────────────┬──────────────────┬────────┬────────┬──────┬──────────────┬─────────┐
            # │ agent_id ┆ dim_0_candidate  ┆ dim_1_candidate  ┆ sugar  ┆ radius ┆ rank ┆ current_rank │ lottery │
            # │ ---      ┆ ---              ┆ ---              ┆ ---    ┆ ---    ┆ ---  ┆ ---          ┆ ---     │
            # │ u64      ┆ i64              ┆ i64              ┆ i64    ┆ i64    ┆ u32  ┆ i64          ┆ f64     │
            # ╞══════════╪══════════════════╪══════════════════╪════════╪════════╪══════╪══════════════╪═════════╡
            winners = (
                best_candidates.sort(["dim_0_candidate", "dim_1_candidate", "lottery"])
                .group_by(["dim_0_candidate", "dim_1_candidate"], maintain_order=True)
                .first()
            )

            assigned = pl.concat(
                [
                    assigned,
                    winners.select(
                        [
                            "agent_id",
                            pl.col("dim_0_candidate"),
                            pl.col("dim_1_candidate"),
                        ]
                    ),
                ],
                how="vertical",
            )
            taken = pl.concat(
                [
                    taken,
                    winners.select(["dim_0_candidate", "dim_1_candidate"]),
                ],
                how="vertical",
            )
            # Origins of agents that move away become available to others in
            # subsequent rounds. Keep origins for agents that stayed put.
            vacated = (
                winners.join(origins_for_filter, on="agent_id", how="left")
                .filter(
                    (pl.col("dim_0_candidate") != pl.col("dim_0_origin"))
                    | (pl.col("dim_1_candidate") != pl.col("dim_1_origin"))
                )
                .select(
                    pl.col("dim_0_origin").alias("dim_0_candidate"),
                    pl.col("dim_1_origin").alias("dim_1_candidate"),
                )
            )
            if not vacated.is_empty():
                taken = taken.join(
                    vacated,
                    on=["dim_0_candidate", "dim_1_candidate"],
                    how="anti",
                )

            winner_ids = winners.select("agent_id")
            unresolved = unresolved.join(winner_ids, on="agent_id", how="anti")
            if unresolved.is_empty():
                break

            # loser candidates columns mirror best_candidates (minus winners).
            losers = best_candidates.join(winner_ids, on="agent_id", how="anti")
            if losers.is_empty():
                continue

            # loser_updates columns (after select):
            # ┌──────────┬───────────┐
            # │ agent_id ┆ next_rank │
            # │ ---      ┆ ---       │
            # │ u64      ┆ i64       │
            # ╞══════════╪═══════════╡
            loser_updates = (
                losers.select(
                    "agent_id",
                    (pl.col("rank") + 1).cast(pl.Int64).alias("next_rank"),
                )
                .join(max_rank, on="agent_id", how="left")
                .with_columns(
                    pl.min_horizontal(pl.col("next_rank"), pl.col("max_rank")).alias(
                        "next_rank"
                    )
                )
                .select(["agent_id", "next_rank"])
            )

            # Promote losers' current_rank (if any) and continue.
            # unresolved (updated) retains columns agent_id/current_rank.
            unresolved = (
                unresolved.join(loser_updates, on="agent_id", how="left")
                .with_columns(
                    pl.when(pl.col("next_rank").is_not_null())
                    .then(pl.col("next_rank"))
                    .otherwise(pl.col("current_rank"))
                    .alias("current_rank")
                )
                .drop("next_rank")
            )

        return assigned


# %% [markdown]
"""
## 4. Run the Model Variants

We iterate over each movement policy with a shared helper so all runs reuse the same seed. The tutorial runs all three variants (Python sequential, Numba sequential, and parallel) by default; edit the script if you want to skip the slow pure-Python baseline.

"""

# %%

GRID_WIDTH = 20
GRID_HEIGHT = 20
NUM_AGENTS = 100
MODEL_STEPS = 60
MAX_SUGAR = 4
SEED = 42


def run_variant(
    agent_cls: type[AntsBase],
    *,
    steps: int,
    seed: int,
) -> tuple[Sugarscape, float]:
    model = Sugarscape(
        agent_type=agent_cls,
        n_agents=NUM_AGENTS,
        width=GRID_WIDTH,
        height=GRID_HEIGHT,
        max_sugar=MAX_SUGAR,
        seed=seed,
    )
    start = perf_counter()
    model.run(steps)
    return model, perf_counter() - start


variant_specs: dict[str, type[AntsBase]] = {
    "Sequential (Python loop)": AntsSequential,
    "Sequential (Numba)": AntsNumba,
    "Parallel (Polars)": AntsParallel,
}

models: dict[str, Sugarscape] = {}
frames: dict[str, pl.DataFrame] = {}
runtimes: dict[str, float] = {}

for variant_name, agent_cls in variant_specs.items():
    model, runtime = run_variant(agent_cls, steps=MODEL_STEPS, seed=SEED)
    models[variant_name] = model
    frames[variant_name] = model.datacollector.data["model"]
    runtimes[variant_name] = runtime

    show_output(
        frames[variant_name]
        .select(["step", "mean_sugar", "total_sugar", "agents_alive"])
        .tail(5),
        title=f"{variant_name} aggregate trajectory (last 5 steps)",
        max_rows=5,
        collapsible=True,
    )
    show_output(f"{variant_name} runtime: {runtime:.3f} s")
    if not _in_ipython():
        print()

runtime_table = (
    pl.DataFrame(
        [
            {
                "update_rule": variant_name,
                "runtime_seconds": runtimes.get(variant_name, float("nan")),
            }
            for variant_name in variant_specs.keys()
        ]
    )
    .with_columns(pl.col("runtime_seconds").round(4))
    .sort("runtime_seconds", descending=False, nulls_last=True)
)

show_output(
    runtime_table,
    title="Runtime comparison (fastest first)",
    collapsible=True,
    open_by_default=True,
)

# Access models/frames on demand; keep namespace minimal.
numba_model_frame = frames.get("Sequential (Numba)", pl.DataFrame())
par_model_frame = frames.get("Parallel (Polars)", pl.DataFrame())


# %% [markdown]
"""
## 5. Comparing the Update Rules

Even though micro rules differ, aggregate trajectories remain qualitatively similar (sugar trends up while population gradually declines).
When we join the traces step-by-step, we see small but noticeable deviations introduced by synchronous conflict resolution (e.g., a few more retirements when conflicts cluster).

In our run (seed=42, with vacated origins available to others), the final-step Gini differs by ≈0.045, and wealth-trait correlations diverge by a few 1e-2 to 1e-1.
These gaps vary by seed and grid size. In practice the parallel rule is best seen as a *performance-oriented relaxation* of the sequential schedule: it can preserve broad macro trends, but it is not a drop-in replacement when you care about seed-level microstructure."""

# %%
comparison = numba_model_frame.select(
    ["step", "mean_sugar", "total_sugar", "agents_alive"]
).join(
    par_model_frame.select(["step", "mean_sugar", "total_sugar", "agents_alive"]),
    on="step",
    how="inner",
    suffix="_parallel",
)
comparison = comparison.with_columns(
    (pl.col("mean_sugar") - pl.col("mean_sugar_parallel")).abs().alias("mean_diff"),
    (pl.col("total_sugar") - pl.col("total_sugar_parallel")).abs().alias("total_diff"),
    (pl.col("agents_alive") - pl.col("agents_alive_parallel"))
    .abs()
    .alias("count_diff"),
)
show_output(
    comparison.select(["step", "mean_diff", "total_diff", "count_diff"]).head(10),
    title="Step-level absolute differences (first 10 steps)",
    max_rows=10,
    collapsible=True,
)


# Build the steady-state metrics table from the DataCollector output rather than
# recomputing reporters directly on the model objects. The collector already
# stored the model-level reporters (gini, correlations, etc.) every step.
def _last_row(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    # Ensure we take the final time step in case steps < MODEL_STEPS due to extinction.
    return df.sort("step").tail(1)


numba_last = _last_row(frames.get("Sequential (Numba)", pl.DataFrame()))
parallel_last = _last_row(frames.get("Parallel (Polars)", pl.DataFrame()))

metrics_pieces: list[pl.DataFrame] = []
if not numba_last.is_empty():
    metrics_pieces.append(
        numba_last.select(
            [
                pl.lit("Sequential (Numba)").alias("update_rule"),
                "gini",
                "corr_sugar_metabolism",
                "corr_sugar_vision",
                pl.col("agents_alive"),
            ]
        )
    )
if not parallel_last.is_empty():
    metrics_pieces.append(
        parallel_last.select(
            [
                pl.lit("Parallel (random tie-break)").alias("update_rule"),
                "gini",
                "corr_sugar_metabolism",
                "corr_sugar_vision",
                pl.col("agents_alive"),
            ]
        )
    )

metrics_table = (
    pl.concat(metrics_pieces, how="vertical") if metrics_pieces else pl.DataFrame()
)

show_output(
    metrics_table.select(
        [
            "update_rule",
            pl.col("gini").round(4),
            pl.col("corr_sugar_metabolism").round(4),
            pl.col("corr_sugar_vision").round(4),
            pl.col("agents_alive"),
        ]
    ),
    title="Steady-state inequality metrics",
    collapsible=True,
    open_by_default=True,
)

if metrics_table.height >= 2:
    numba_gini = metrics_table.filter(pl.col("update_rule") == "Sequential (Numba)")[
        "gini"
    ][0]
    par_gini = metrics_table.filter(
        pl.col("update_rule") == "Parallel (random tie-break)"
    )["gini"][0]
    show_output(
        f"Absolute Gini gap (numba vs parallel): {abs(numba_gini - par_gini):.4f}"
    )

# %% [markdown]
"""
## 6. Takeaways and Next Steps

Some final notes:
- mesa-frames should preferably be used when you have many agents and operations can be vectorized.
- If your model is not easily vectorizable, consider using Numba or reducing your microscopic rule to a vectorizable form. As we saw, the macroscopic behavior can remain consistent (and be more similar to real-world systems).


Currently, the Polars implementation spends most of the time in join operations.

**Polars + LazyFrames roadmap** - future mesa-frames releases will expose
  LazyFrame-powered sets and spaces (which can also use a GPU cuda accelerated backend which greatly accelerates joins), so the same Polars
  code you wrote here will scale even further without touching Numba.
"""
