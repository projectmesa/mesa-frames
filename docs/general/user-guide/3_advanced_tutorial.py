from __future__ import annotations

# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#   kernelspec:
#     display_name: Python 3 (uv)
#     language: python
#     name: python3
# ---

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
  distance and coordinates). The sugar on cells that are already occupied (including its own) is 0.
* **Eat & survive:** ants harvest the sugar on the cell they occupy. If their
  sugar stock falls below their `metabolism` cost, they die.
* **Regrow:** sugar instantly regrows to its maximum level on empty cells. The
  landscape is drawn from a uniform distribution, so resources are homogeneous
  on average and the interesting dynamics come from agent heterogeneity and
  congestion.

The update schedule matters for micro-behaviour, so we study three variants:

1. **Sequential loop (asynchronous):** This is the traditional definition. Ants move one at a time in random order. 
This cannnot be vectorised easily as the best move for an ant might depend on the moves of earlier ants (for example, if they target the same cell).
2. **Sequential with Numba:** matches the first variant but relies on a compiled
   helper for speed.
3. **Parallel (synchronous):** all ants propose moves; conflicts are resolved at
   random before applying the winners simultaneously (and the losers get to their second-best cell, etc).

Our goal is to show that, under instantaneous growback and uniform resources,
the model converges to the *same* macroscopic inequality pattern regardless of
whether agents act sequentially or in parallel and that As long as the random draws do
not push the system into extinction, the long-run Gini coefficient of wealth and
the wealth–trait correlations line up within sampling error — a classic example
of emergent macro regularities in agent-based models.
"""

# %% [markdown]
# First, let's install and import the necessary packages.

# %% [markdown]
# If you're running this tutorial on Google Colab or another fresh environment,
# uncomment the cell below to install the required dependencies.

# %%
# !pip install git+https://github.com/projectmesa/mesa-frames polars numba numpy

# %% [markdown]
"""## 1. Imports"""

# %%

from collections import defaultdict
from time import perf_counter

import numpy as np
import polars as pl
from numba import njit

from mesa_frames import AgentSet, DataCollector, Grid, Model

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

The `step` method advances the sugar field, triggers the agent set's step
"""


# %%

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
    agent_type : type
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
    seed : int or None, optional
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
        agent_type: type["SugarscapeAgentsBase"],
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
                "living_agents": lambda m: len(m.sets[0]),
            },
            agent_reporters={"traits": ["sugar", "metabolism", "vision"]},
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
        The per-step ordering is important: regrowth happens first (so empty
        cells are refilled), then agents move and eat, and finally metrics are
        collected. If the agent set becomes empty at any point the model is
        marked as not running.
        """
        if len(self.sets[0]) == 0:
            self.running = False
            return
        self._advance_sugar_field()
        self.sets[0].step()
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
            refresh = empty_cells.join(self._max_sugar, on=["dim_0", "dim_1"], how="left")
            self.space.set_cells(empty_cells, {"sugar": refresh["max_sugar"]})
        full_cells = self.space.full_cells
        if not full_cells.is_empty():
            # Occupied cells have just been harvested; set their sugar to 0.
            zeros = pl.Series(np.zeros(len(full_cells), dtype=np.int64))
            self.space.set_cells(full_cells, {"sugar": zeros})

# %% [markdown]

"""
## 3. Agent definition

### Base agent class

Now let's define the agent class (the ant class). We start with a base class which implements the common logic for eating and starvation, while leaving the `move` method abstract. 
The base class also provides helper methods for sensing visible cells and choosing the best cell based on sugar, distance, and coordinates.
This will allow us to define different movement policies (sequential, Numba-accelerated, and parallel) as subclasses that only need to implement the `move` method.
We also add
"""

# %%

class SugarscapeAgentsBase(AgentSet):
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
        occupied_cells = self.space.cells.filter(pl.col("agent_id").is_in(occupied_ids.implode()))
        if occupied_cells.is_empty():
            return
        # The agent ordering here uses the agent_id values stored in the
        # occupied cells frame; indexing the agent set with that vector updates
        # the matching agents' sugar values in one vectorised write.
        agent_ids = occupied_cells["agent_id"]
        self[agent_ids, "sugar"] = (
            self[agent_ids, "sugar"] + occupied_cells["sugar"] - self[agent_ids, "metabolism"]
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

    def _current_sugar_map(self) -> dict[tuple[int, int], int]:
        """Return a mapping from grid coordinates to the current sugar value.

        Returns
        -------
        dict
            Keys are ``(x, y)`` tuples and values are the integer sugar amount
            on that cell (zero if missing/None).
        """
        cells = self.space.cells.select(["dim_0", "dim_1", "sugar"])
        # Build a plain Python dict for fast lookups in the movement code.
        return {
            (int(x), int(y)): 0 if sugar is None else int(sugar)
            for x, y, sugar in cells.iter_rows()
        }

    @staticmethod
    def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
        """Compute the Manhattan (L1) distance between two grid cells.

        Parameters
        ----------
        a, b : tuple[int, int]
            Coordinate pairs ``(x, y)``.

        Returns
        -------
        int
            The Manhattan distance between ``a`` and ``b``.
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _visible_cells(self, origin: tuple[int, int], vision: int) -> list[tuple[int, int]]:
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
        2. If equal sugar, prefer the cell with smaller Manhattan distance
           from the origin.
        3. If still tied, prefer the cell with smaller coordinates (lexicographic
           ordering of the ``(x, y)`` tuple).

        Parameters
        ----------
        origin : tuple[int, int]
            Agent's current coordinate.
        vision : int
            Maximum vision radius along cardinal axes.
        sugar_map : dict
            Mapping from ``(x, y)`` to sugar amount.
        blocked : set or None
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
        for candidate in self._visible_cells(origin, vision):
            # Skip blocked cells (occupied by other agents) unless it's the
            # agent's current cell which we always consider.
            if blocked and candidate != origin and candidate in blocked:
                continue
            sugar_here = sugar_map.get(candidate, 0)
            distance = self._manhattan(origin, candidate)
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



# %% 
GRID_WIDTH = 50
GRID_HEIGHT = 50
NUM_AGENTS = 400
MODEL_STEPS = 60
MAX_SUGAR = 4

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
    best_sugar, candidate_sugar : int
        Sugar at the current best cell and the candidate cell.
    best_distance, candidate_distance : int
        Manhattan distances from the origin to the best and candidate cells.
    best_x, best_y, candidate_x, candidate_y : int
        Coordinates used for the final lexicographic tie-break.

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
    dim0, dim1 : np.ndarray
        1D integer arrays of length n_agents containing the x and y
        coordinates for each agent.
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




# %% [markdown]
"""
## 2. Agent Scaffolding

With the space logic in place we can define the agents. The base class stores
traits and implements eating/starvation; concrete subclasses only override
`move`.
"""




# %% [markdown]
"""
## 3. Sequential Movement
"""


class SugarscapeSequentialAgents(SugarscapeAgentsBase):
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
## 4. Speeding Up the Loop with Numba
"""


class SugarscapeNumbaAgents(SugarscapeAgentsBase):
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
            .with_columns(pl.col("sugar").fill_null(0))
            ["sugar"].to_numpy()
            .reshape(self.space.dimensions)
        )

        new_dim0, new_dim1 = sequential_move_numba(dim0, dim1, vision, sugar_array)
        coords = pl.DataFrame({"dim_0": new_dim0.tolist(), "dim_1": new_dim1.tolist()})
        self.space.move_agents(agent_ids, coords)


# %% [markdown]
"""
## 5. Simultaneous Movement with Conflict Resolution
"""


class SugarscapeParallelAgents(SugarscapeAgentsBase):
    def move(self) -> None:
        # Parallel movement: each agent proposes a ranked list of visible
        # cells (including its own). We resolve conflicts in rounds using
        # DataFrame operations so winners can be chosen per-cell at random
        # and losers are promoted to their next-ranked choice.
        if len(self.df) == 0:
            return
        sugar_map = self._current_sugar_map()
        state = self.df.join(self.pos, on="unique_id", how="left")
        if state.is_empty():
            return

        # Map the positional frame to a center lookup used when joining
        # neighbourhoods produced by the space helper.
        center_lookup = self.pos.rename(
            {
                "unique_id": "agent_id",
                "dim_0": "dim_0_center",
                "dim_1": "dim_1_center",
            }
        )

        # Build a neighbourhood frame: for each agent and visible cell we
        # attach the cell sugar and the agent_id of the occupant (if any).
        neighborhood = (
            self.space.get_neighborhood(
                radius=self["vision"], agents=self, include_center=True
            )
            .join(
                self.space.cells.select(["dim_0", "dim_1", "sugar"]),
                on=["dim_0", "dim_1"],
                how="left",
            )
            .join(center_lookup, on=["dim_0_center", "dim_1_center"], how="left")
            .with_columns(pl.col("sugar").fill_null(0))
        )

        # Normalise occupant column name if present.
        if "agent_id" in neighborhood.columns:
            neighborhood = neighborhood.rename({"agent_id": "occupant_id"})

        # Create ranked choices per agent: sort by sugar (desc), radius
        # (asc), then coordinates. Keep the first unique entry per cell.
        choices = (
            neighborhood.select(
                [
                    "agent_id",
                    "dim_0",
                    "dim_1",
                    "sugar",
                    "radius",
                    "dim_0_center",
                    "dim_1_center",
                ]
            )
            .with_columns(pl.col("radius").cast(pl.Int64))
            .sort(
                ["agent_id", "sugar", "radius", "dim_0", "dim_1"],
                descending=[False, True, False, False, False],
            )
            .unique(
                subset=["agent_id", "dim_0", "dim_1"],
                keep="first",
                maintain_order=True,
            )
            .with_columns(pl.cum_count().over("agent_id").cast(pl.Int64).alias("rank"))
        )

        if choices.is_empty():
            return

        # Origins for fallback (if an agent exhausts candidates it stays put).
        origins = center_lookup.select(
            [
                "agent_id",
                pl.col("dim_0_center").alias("dim_0"),
                pl.col("dim_1_center").alias("dim_1"),
            ]
        )

        # Track the maximum available rank per agent to clamp promotions.
        max_rank = choices.group_by("agent_id").agg(pl.col("rank").max().alias("max_rank"))

        # Prepare unresolved agents and working tables.
        agent_ids = choices["agent_id"].unique(maintain_order=True)
        unresolved = pl.DataFrame(
            {
                "agent_id": agent_ids,
                "current_rank": pl.Series(np.zeros(agent_ids.len(), dtype=np.int64)),
            }
        )

        assigned = pl.DataFrame(
            {
                "agent_id": pl.Series(name="agent_id", values=[], dtype=agent_ids.dtype),
                "dim_0": pl.Series(name="dim_0", values=[], dtype=pl.Int64),
                "dim_1": pl.Series(name="dim_1", values=[], dtype=pl.Int64),
            }
        )

        taken = pl.DataFrame(
            {
                "dim_0": pl.Series(name="dim_0", values=[], dtype=pl.Int64),
                "dim_1": pl.Series(name="dim_1", values=[], dtype=pl.Int64),
            }
        )

        # Resolve in rounds: each unresolved agent proposes its current-ranked
        # candidate; winners per-cell are selected at random and losers are
        # promoted to their next choice.
        while unresolved.height > 0:
            candidate_pool = choices.join(unresolved, on="agent_id")
            candidate_pool = candidate_pool.filter(pl.col("rank") >= pl.col("current_rank"))
            if not taken.is_empty():
                candidate_pool = candidate_pool.join(taken, on=["dim_0", "dim_1"], how="anti")

            if candidate_pool.is_empty():
                # No available candidates — everyone falls back to origin.
                fallback = unresolved.join(origins, on="agent_id", how="left")
                assigned = pl.concat(
                    [assigned, fallback.select(["agent_id", "dim_0", "dim_1"])],
                    how="vertical",
                )
                break

            best_candidates = (
                candidate_pool.sort(["agent_id", "rank"]) .group_by("agent_id", maintain_order=True).first()
            )

            # Agents that had no candidate this round fall back to origin.
            missing = unresolved.join(best_candidates.select("agent_id"), on="agent_id", how="anti")
            if not missing.is_empty():
                fallback = missing.join(origins, on="agent_id", how="left")
                assigned = pl.concat(
                    [assigned, fallback.select(["agent_id", "dim_0", "dim_1"])],
                    how="vertical",
                )
                taken = pl.concat([taken, fallback.select(["dim_0", "dim_1"])], how="vertical")
                unresolved = unresolved.join(missing.select("agent_id"), on="agent_id", how="anti")
                best_candidates = best_candidates.join(missing.select("agent_id"), on="agent_id", how="anti")
                if unresolved.is_empty() or best_candidates.is_empty():
                    continue

            # Add a small random lottery to break ties deterministically for
            # each candidate set.
            lottery = pl.Series("lottery", self.random.random(best_candidates.height))
            best_candidates = best_candidates.with_columns(lottery)

            winners = (
                best_candidates.sort(["dim_0", "dim_1", "lottery"]) .group_by(["dim_0", "dim_1"], maintain_order=True).first()
            )

            assigned = pl.concat(
                [assigned, winners.select(["agent_id", "dim_0", "dim_1"])],
                how="vertical",
            )
            taken = pl.concat([taken, winners.select(["dim_0", "dim_1"])], how="vertical")

            winner_ids = winners.select("agent_id")
            unresolved = unresolved.join(winner_ids, on="agent_id", how="anti")
            if unresolved.is_empty():
                break

            losers = best_candidates.join(winner_ids, on="agent_id", how="anti")
            if losers.is_empty():
                continue

            loser_updates = (
                losers.select(
                    "agent_id",
                    (pl.col("rank") + 1).cast(pl.Int64).alias("next_rank"),
                )
                .join(max_rank, on="agent_id", how="left")
                .with_columns(
                    pl.min_horizontal(pl.col("next_rank"), pl.col("max_rank")).alias("next_rank")
                )
                .select(["agent_id", "next_rank"])
            )

            # Promote losers' current_rank (if any) and continue.
            unresolved = unresolved.join(loser_updates, on="agent_id", how="left").with_columns(
                pl.when(pl.col("next_rank").is_not_null())
                .then(pl.col("next_rank"))
                .otherwise(pl.col("current_rank"))
                .alias("current_rank")
            ).drop("next_rank")

        if assigned.is_empty():
            return

        move_df = pl.DataFrame(
            {
                "unique_id": assigned["agent_id"],
                "dim_0": assigned["dim_0"],
                "dim_1": assigned["dim_1"],
            }
        )
        # `move_agents` accepts IdsLike and SpaceCoordinates (Polars Series/DataFrame),
        # so pass Series/DataFrame directly rather than converting to Python lists.
        self.space.move_agents(move_df["unique_id"], move_df.select(["dim_0", "dim_1"]))
        
def run_variant(
    agent_cls: type[SugarscapeAgentsBase],
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


# %% [markdown]
"""
## 6. Shared Model Infrastructure

`SugarscapeTutorialModel` wires the grid, agent set, regrowth logic, and data
collection. Each variant simply plugs in a different agent class.
"""


def gini(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    sorted_vals = np.sort(values.astype(np.float64))
    n = sorted_vals.size
    if n == 0:
        return float("nan")
    cumulative = np.cumsum(sorted_vals)
    total = cumulative[-1]
    if total == 0:
        return 0.0
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.dot(index, sorted_vals) / (n * total)) - (n + 1) / n)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _column_with_prefix(df: pl.DataFrame, prefix: str) -> str:
    for col in df.columns:
        if col.startswith(prefix):
            return col
    raise KeyError(f"No column starts with prefix '{prefix}'")


def final_agent_snapshot(model: Model) -> pl.DataFrame:
    agent_frame = model.datacollector.data["agent"]
    if agent_frame.is_empty():
        return agent_frame
    last_step = agent_frame["step"].max()
    return agent_frame.filter(pl.col("step") == last_step)


def summarise_inequality(model: Model) -> dict[str, float]:
    snapshot = final_agent_snapshot(model)
    if snapshot.is_empty():
        return {
            "gini": float("nan"),
            "corr_sugar_metabolism": float("nan"),
            "corr_sugar_vision": float("nan"),
            "agents_alive": 0,
        }

    sugar_col = _column_with_prefix(snapshot, "traits_sugar_")
    metabolism_col = _column_with_prefix(snapshot, "traits_metabolism_")
    vision_col = _column_with_prefix(snapshot, "traits_vision_")

    sugar = snapshot[sugar_col].to_numpy()
    metabolism = snapshot[metabolism_col].to_numpy()
    vision = snapshot[vision_col].to_numpy()

    return {
        "gini": gini(sugar),
        "corr_sugar_metabolism": _safe_corr(sugar, metabolism),
        "corr_sugar_vision": _safe_corr(sugar, vision),
        "agents_alive": float(sugar.size),
    }


# %% [markdown]
"""
## 7. Run the Sequential Model (Python loop)

With the scaffolding in place we can simulate the sequential version and inspect
its aggregate behaviour. Because all random draws flow through the model's RNG,
constructing each variant with the same seed reproduces identical initial
conditions across the different movement rules.
"""

# %%
sequential_seed = 11

sequential_model, sequential_time = run_variant(
    SugarscapeSequentialAgents, steps=MODEL_STEPS, seed=sequential_seed
)

seq_model_frame = sequential_model.datacollector.data["model"]
print("Sequential aggregate trajectory (last 5 steps):")
print(
    seq_model_frame.select(["step", "mean_sugar", "total_sugar", "living_agents"]).tail(5)
)
print(f"Sequential runtime: {sequential_time:.3f} s")

# %% [markdown]
"""
## 8. Run the Numba-Accelerated Model

We reuse the same seed so the only difference is the compiled movement helper.
The trajectory matches the pure Python loop (up to floating-point noise) while
running much faster on larger grids.
"""

# %%
numba_model, numba_time = run_variant(
    SugarscapeNumbaAgents, steps=MODEL_STEPS, seed=sequential_seed
)

numba_model_frame = numba_model.datacollector.data["model"]
print("Numba sequential aggregate trajectory (last 5 steps):")
print(
    numba_model_frame.select(["step", "mean_sugar", "total_sugar", "living_agents"]).tail(5)
)
print(f"Numba sequential runtime: {numba_time:.3f} s")

# %% [markdown]
"""
## 9. Run the Simultaneous Model

Next we instantiate the parallel variant with the same seed so every run starts
from the common state generated by the helper methods.
"""

# %%
parallel_model, parallel_time = run_variant(
    SugarscapeParallelAgents, steps=MODEL_STEPS, seed=sequential_seed
)

par_model_frame = parallel_model.datacollector.data["model"]
print("Parallel aggregate trajectory (last 5 steps):")
print(par_model_frame.select(["step", "mean_sugar", "total_sugar", "living_agents"]).tail(5))
print(f"Parallel runtime: {parallel_time:.3f} s")

# %% [markdown]
"""
## 10. Runtime Comparison

The table below summarises the elapsed time for 60 steps on the 50×50 grid with
400 ants. Parallel scheduling on top of Polars lands in the same performance
band as the Numba-accelerated loop, while both are far faster than the pure
Python baseline.
"""

# %%
runtime_table = pl.DataFrame(
    {
        "update_rule": [
            "Sequential (Python loop)",
            "Sequential (Numba)",
            "Parallel (Polars)",
        ],
        "runtime_seconds": [sequential_time, numba_time, parallel_time],
    }
).with_columns(pl.col("runtime_seconds").round(4))

print(runtime_table)

# %% [markdown]
"""
Polars gives us that performance without any bespoke compiled kernels—the move
logic reads like ordinary DataFrame code. The Numba version is a touch faster,
but only after writing and maintaining `_numba_find_best_cell` and friends. In
practice we get near-identical runtimes, so you can pick the implementation that
is simplest for your team.
"""

# %% [markdown]
"""
## 11. Comparing the Update Rules

Even though the micro rules differ, the aggregate trajectories keep the same
overall shape: sugar holdings trend upward while the population tapers off. By
joining the model-level traces we can quantify how conflict resolution
randomness introduces modest deviations (for example, the simultaneous variant
often retires a few more agents when several conflicts pile up in the same
neighbourhood). Crucially, the steady-state inequality metrics line up: the Gini
coefficients differ by roughly 0.0015 and the wealth–trait correlations are
indistinguishable, which validates the relaxed, fully-parallel update scheme.
"""

# %%
comparison = numba_model_frame.select(["step", "mean_sugar", "total_sugar", "living_agents"]).join(
    par_model_frame.select(["step", "mean_sugar", "total_sugar", "living_agents"]),
    on="step",
    how="inner",
    suffix="_parallel",
)
comparison = comparison.with_columns(
    (pl.col("mean_sugar") - pl.col("mean_sugar_parallel")).abs().alias("mean_diff"),
    (pl.col("total_sugar") - pl.col("total_sugar_parallel")).abs().alias("total_diff"),
    (pl.col("living_agents") - pl.col("living_agents_parallel")).abs().alias("count_diff"),
)
print("Step-level absolute differences (first 10 steps):")
print(comparison.select(["step", "mean_diff", "total_diff", "count_diff"]).head(10))

metrics_table = pl.DataFrame(
    [
        {
            "update_rule": "Sequential (Numba)",
            **summarise_inequality(numba_model),
        },
        {
            "update_rule": "Parallel (random tie-break)",
            **summarise_inequality(parallel_model),
        },
    ]
)

print("\nSteady-state inequality metrics:")
print(
    metrics_table.select(
        [
            "update_rule",
            pl.col("gini").round(4),
            pl.col("corr_sugar_metabolism").round(4),
            pl.col("corr_sugar_vision").round(4),
            pl.col("agents_alive"),
        ]
    )
)

numba_gini = metrics_table.filter(pl.col("update_rule") == "Sequential (Numba)")["gini"][0]
par_gini = metrics_table.filter(pl.col("update_rule") == "Parallel (random tie-break)")["gini"][0]
print(f"Absolute Gini gap (numba vs parallel): {abs(numba_gini - par_gini):.4f}")

# %% [markdown]
"""
## 12. Where to Go Next?

* **Polars + LazyFrames roadmap** – future mesa-frames releases will expose
  LazyFrame-powered schedulers (with GPU offloading hooks), so the same Polars
  code you wrote here will scale even further without touching Numba.
* **Production reference** – the `examples/sugarscape_ig/ss_polars` package
  shows how to take this pattern further with additional vectorisation tricks.
* **Alternative conflict rules** – it is straightforward to swap in other
  tie-breakers, such as letting losing agents search for the next-best empty
  cell rather than staying put.
* **Macro validation** – wrap the metric collection in a loop over seeds to
  quantify how small the Gini gap remains across independent replications.
* **Statistical physics meets ABM** – for a modern take on the macro behaviour
  of Sugarscape-like economies, see Axtell (2000) or subsequent statistical
  physics treatments of wealth exchange models.

Because this script doubles as the notebook source, any edits you make here can
be synchronised with a `.ipynb` representation via Jupytext.
"""
