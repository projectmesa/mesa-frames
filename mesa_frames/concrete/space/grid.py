"""
Polars-based implementation of spatial structures for mesa-frames.

This module provides concrete implementations of spatial structures using Polars
as the backend for DataFrame operations. It defines the Grid class, which
implements a 2D grid structure using Polars DataFrames for efficient spatial
operations and agent positioning.

Classes:
    Grid(AbstractGrid, PolarsMixin):
        A Polars-based implementation of a 2D grid. This class uses Polars
        DataFrames to store and manipulate spatial data, providing high-performance
        operations for large-scale spatial simulations.

The Grid class is designed to be used within Model instances to represent
the spatial environment of the simulation. It leverages the power of Polars for
fast and efficient data operations on spatial attributes and agent positions.

Usage:
    The Grid class can be used directly in a model to represent the
    spatial environment:

    from mesa_frames.concrete.model import Model
    from mesa_frames.concrete.space import Grid
    from mesa_frames.concrete.agentset import AgentSet

    class MyAgents(AgentSet):
        # ... agent implementation ...

    class MyModel(Model):
        def __init__(self, width, height):
            super().__init__()
            self.space = Grid(self, [width, height])
            self.sets += MyAgents(self)

        def step(self):
            # Move agents
            self.space.move_agents(self.sets)
            # ... other model logic ...

For more detailed information on the Grid class and its methods,
refer to the class docstring.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from itertools import product
import os
from warnings import warn

import mesa_frames
import numpy as np
import polars as pl

from mesa_frames.abstract.space import AbstractGrid
from mesa_frames.abstract.agentset import AbstractAgentSet
from mesa_frames.abstract.agentsetregistry import AbstractAgentSetRegistry
from .cells import GridCells
from .neighborhood import GridNeighborhood
from mesa_frames.concrete.mixin import PolarsMixin
from mesa_frames.utils import copydoc
from mesa_frames.types_ import (
    ArrayLike,
    DataFrame,
    GridCoordinate,
    GridCoordinates,
    IdsLike,
    Series,
)


@copydoc(AbstractGrid)
class Grid(AbstractGrid, PolarsMixin):
    """Polars-based implementation of AbstractGrid."""

    _agents: pl.DataFrame
    _copy_with_method: dict[str, tuple[str, list[str]]] = {
        "_agents": ("clone", []),
        "_offsets": ("clone", []),
    }
    _offsets: pl.DataFrame

    def __init__(
        self,
        model: mesa_frames.concrete.model.Model,
        dimensions: Sequence[int],
        torus: bool = False,
        capacity: int | None = None,
        neighborhood_type: str = "moore",
    ) -> None:
        # Call the next __init__ after AbstractGrid (AbstractDiscreteSpace)
        # without invoking AbstractGrid's interface-only __init__.
        super(AbstractGrid, self).__init__(model=model, capacity=capacity)

        self._dimensions = dimensions
        self._torus = torus
        self._pos_col_names = [f"dim_{k}" for k in range(len(dimensions))]
        self._center_col_names = [x + "_center" for x in self._pos_col_names]
        self._agents = self._df_constructor(
            columns=["agent_id"] + self._pos_col_names,
            index_cols="agent_id",
            dtypes={"agent_id": "uint64"} | {col: int for col in self._pos_col_names},
        )
        self._offsets = self._compute_offsets(neighborhood_type)
        self._neighborhood_type = neighborhood_type

        self.cells = GridCells(self)
        self.neighborhood = GridNeighborhood(self)

        from mesa_frames.concrete.space._grid_fastpath import _GridFastPath

        self._fastpath = _GridFastPath(self)

    @property
    def cells(self) -> GridCells:
        return self._cells_obj

    @cells.setter
    def cells(self, cells: GridCells) -> None:
        self._cells_obj = cells

    @property
    def neighborhood(self) -> GridNeighborhood:
        return self._neighborhood_obj

    @neighborhood.setter
    def neighborhood(self, neighborhood: GridNeighborhood) -> None:
        self._neighborhood_obj = neighborhood

    def get_directions(
        self,
        pos0: GridCoordinate | GridCoordinates | None = None,
        pos1: GridCoordinate | GridCoordinates | None = None,
        agents0: IdsLike
        | AbstractAgentSet
        | AbstractAgentSetRegistry
        | Collection[AbstractAgentSet]
        | Collection[AbstractAgentSetRegistry]
        | None = None,
        agents1: IdsLike
        | AbstractAgentSet
        | AbstractAgentSetRegistry
        | Collection[AbstractAgentSet]
        | Collection[AbstractAgentSetRegistry]
        | None = None,
        normalize: bool = False,
    ) -> DataFrame:
        result = self._calculate_differences(pos0, pos1, agents0, agents1)
        if normalize:
            result = self._df_div(result, other=self._df_norm(result))
        return result

    def get_distances(
        self,
        pos0: GridCoordinate | GridCoordinates | None = None,
        pos1: GridCoordinate | GridCoordinates | None = None,
        agents0: IdsLike
        | AbstractAgentSet
        | AbstractAgentSetRegistry
        | Collection[AbstractAgentSet]
        | Collection[AbstractAgentSetRegistry]
        | None = None,
        agents1: IdsLike
        | AbstractAgentSet
        | AbstractAgentSetRegistry
        | Collection[AbstractAgentSet]
        | Collection[AbstractAgentSetRegistry]
        | None = None,
    ) -> DataFrame:
        result = self._calculate_differences(pos0, pos1, agents0, agents1)
        return self._df_norm(result, "distance", True)

    def out_of_bounds(self, pos: GridCoordinate | GridCoordinates) -> DataFrame:
        if self.torus:
            raise ValueError("This method is only valid for non-torus grids")
        pos_df = self._get_df_coords(pos, check_bounds=False)
        out_of_bounds = self._df_all(
            self._df_or(
                pos_df < 0,
                self._df_ge(
                    pos_df,
                    self._dimensions,
                    axis="columns",
                    index_cols=self._pos_col_names,
                ),
            ),
            name="out_of_bounds",
        )
        return self._df_concat(
            objs=[pos_df, self._srs_to_df(out_of_bounds)], how="horizontal"
        )

    def remove_agents(
        self,
        agents: IdsLike
        | AbstractAgentSet
        | AbstractAgentSetRegistry
        | Collection[AbstractAgentSet]
        | Collection[AbstractAgentSetRegistry],
        inplace: bool = True,
    ) -> Grid:
        if not inplace:
            obj = self.copy()
            return obj.remove_agents(agents, inplace=True)

        agents = self._get_ids_srs(agents)
        if __debug__:
            b_contained = agents.is_in(self.model.sets.ids)
            if (isinstance(b_contained, Series) and not b_contained.all()) or (
                isinstance(b_contained, bool) and not b_contained
            ):
                raise ValueError("Some agents are not in the model")

        self.cells.update_capacity_agents(agents, operation="removal")
        self._agents = self._df_remove(self._agents, mask=agents, index_cols="agent_id")
        return self

    def move_all(
        self,
        agents: IdsLike
        | AbstractAgentSet
        | AbstractAgentSetRegistry
        | Collection[AbstractAgentSet]
        | Collection[AbstractAgentSetRegistry],
        pos: GridCoordinate | GridCoordinates,
        inplace: bool = True,
    ) -> Grid:
        if not inplace:
            obj = self.copy()
            return obj.move_all(agents, pos, inplace=True)

        return self._place_or_move_agents(
            agents=agents,
            pos=pos,
            is_move=True,
            trust_full_move=True,
            require_full_move=True,
        )

    def _explain_move_to_best_path(
        self,
        *,
        radius: int | ArrayLike,
        property: str,  # noqa: A002
        include_center: bool,
    ) -> dict[str, object]:
        """Explain which implementation path will be used for move_to_best.

        This is an internal debug hook; it is not part of the public API.
        """
        reasons: list[str] = []

        forced = (
            os.environ.get("MESA_FRAMES_GRID_MOVE_TO_BEST_FORCE_PATH", "")
            .strip()
            .lower()
        )
        if forced in {"fast", "df"}:
            # Preserve compatibility: forced controls the high-level path.
            fast_kind = "numba" if self._fastpath.numba_enabled else "python"
            return {
                "path": forced,
                "fast_kind": fast_kind,
                "reasons": [f"forced via env: {forced}"],
            }

        if len(self._dimensions) != 2:
            reasons.append("dimensions not 2D")
        if self._neighborhood_type not in {"moore", "von_neumann"}:
            reasons.append("neighborhood_type not supported")

        # Radius can be scalar or per-agent (sequence/Series/ndarray). For fast path,
        # we require integer-valued, non-negative radii.
        radius_np: np.ndarray | None = None
        if isinstance(radius, Series):
            radius_np = radius.to_numpy()
        elif isinstance(radius, np.ndarray):
            radius_np = radius
        elif isinstance(radius, Collection) and not isinstance(radius, (str, bytes)):
            radius_np = np.asarray(radius)

        if radius_np is None:
            try:
                radius_int = int(radius)  # type: ignore[arg-type]
            except Exception:
                reasons.append("radius must be a non-negative int or integer sequence")
            else:
                if isinstance(radius, float) and not float(radius).is_integer():
                    reasons.append("radius must be integer-valued")
                if radius_int < 0:
                    reasons.append("radius must be a non-negative int")
        else:
            if radius_np.ndim != 1:
                reasons.append("radius sequence must be 1-D")
            elif radius_np.size == 0:
                reasons.append("radius sequence must be non-empty")
            elif not np.issubdtype(radius_np.dtype, np.integer):
                reasons.append("radius sequence must have integer dtype")
            elif np.any(radius_np < 0):
                reasons.append("radius values must be >= 0")

        # Numba gating: we want to avoid Python loops on the hot path.
        # - per-agent radius requires Numba to be eligible for the fast path
        # - scalar radius also prefers Numba (conflict resolution is iterative)
        if radius_np is not None and not self._fastpath.numba_enabled:
            reasons.append("per-agent radius requires numba for fast path")
        if radius_np is None and not self._fastpath.numba_enabled:
            reasons.append("numba unavailable for fast path")
        if not isinstance(include_center, bool):
            reasons.append("include_center must be a bool")
        if self._agents.is_empty():
            reasons.append("no agents placed")

        # Dense row-major property buffer requirements.
        if self.cells._cells.is_empty():
            reasons.append("cells properties not initialized")
        else:
            if self.cells._cells.height != int(np.prod(self._dimensions)):
                reasons.append("cells not dense")
            if property not in self.cells._cells.columns:
                reasons.append("property column missing")
            else:
                # Only sort/canonicalize when the table is already dense.
                # Do NOT densify here: this function is a debug hook and should not
                # mutate state or change the chosen path.
                if self.cells._cells.height == int(np.prod(self._dimensions)):
                    self.cells._ensure_dense_row_major_cells()
                dtype = self.cells._cells.schema.get(property)
                numeric = dtype in {
                    pl.Int8,
                    pl.Int16,
                    pl.Int32,
                    pl.Int64,
                    pl.UInt8,
                    pl.UInt16,
                    pl.UInt32,
                    pl.UInt64,
                    pl.Float32,
                    pl.Float64,
                }
                if not numeric:
                    reasons.append("property column not numeric")

        fast_kind = "numba" if self._fastpath.numba_enabled else "python"
        if reasons:
            return {"path": "df", "fast_kind": fast_kind, "reasons": reasons}
        return {"path": "fast", "fast_kind": fast_kind, "reasons": ["eligible"]}

    def move_to_best(
        self,
        agents: IdsLike
        | AbstractAgentSet
        | AbstractAgentSetRegistry
        | Collection[AbstractAgentSet]
        | Collection[AbstractAgentSetRegistry],
        radius: int | ArrayLike,
        property: str = "sugar",  # noqa: A002
        include_center: bool = True,
        *,
        inplace: bool = True,
    ) -> Grid:
        """Move agents to the best neighboring cell by a simple cell property.

        Ranking is by:
        - property (descending)
        - radius (ascending)
        - dim_0 (ascending)
        - dim_1 (ascending)

        Conflicts are resolved with a deterministic, seeded round-based lottery.
        """
        if not inplace:
            obj = self.copy()
            return obj.move_to_best(
                agents=agents,
                radius=radius,
                property=property,
                include_center=include_center,
                inplace=True,
            )

        if len(self._dimensions) != 2:
            raise ValueError("move_to_best is only supported for 2D grids")

        move_ids_srs = self._get_ids_srs(agents)
        if move_ids_srs.is_empty() or self._agents.is_empty():
            return self

        # Normalize radius.
        radius_arr: np.ndarray | None = None
        radius_scalar: int | None = None
        radius_aligned_to: str | None = None  # "move" | "placed"

        if isinstance(radius, Series):
            radius_arr = radius.to_numpy()
        elif isinstance(radius, np.ndarray):
            radius_arr = radius
        elif isinstance(radius, Collection) and not isinstance(radius, (str, bytes)):
            radius_arr = np.asarray(radius)
        else:
            try:
                radius_scalar = int(radius)  # type: ignore[arg-type]
            except Exception as e:
                raise TypeError("radius must be an int or an integer sequence") from e

        if radius_arr is not None:
            radius_arr = np.asarray(radius_arr)
            if radius_arr.ndim != 1:
                raise ValueError("radius sequence must be 1-D")
            if radius_arr.size == 0:
                return self
            if not np.issubdtype(radius_arr.dtype, np.integer):
                raise TypeError("radius sequence must have integer dtype")
            if np.any(radius_arr < 0):
                raise ValueError("radius values must be >= 0")
            radius_arr = radius_arr.astype(np.int64, copy=False)

            n_move = int(move_ids_srs.len())
            n_placed = int(self._agents.height)
            if radius_arr.size == n_move:
                radius_aligned_to = "move"
            elif radius_arr.size == n_placed:
                radius_aligned_to = "placed"
            else:
                raise ValueError(
                    "radius sequence length must match the number of agents"
                )
        else:
            if radius_scalar is None:
                raise TypeError("radius must be an int or an integer sequence")
            if isinstance(radius, float) and not float(radius).is_integer():
                raise ValueError("radius must be integer-valued")
            if radius_scalar < 0:
                raise ValueError("radius must be >= 0")

        # Validate property up front.
        if self.cells._cells.is_empty() or property not in self.cells._cells.columns:
            raise ValueError(f"Unknown cell property: {property}")
        dtype = self.cells._cells.schema.get(property)
        numeric = dtype in {
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
            pl.Float32,
            pl.Float64,
        }
        if not numeric:
            raise ValueError(f"Cell property must be numeric: {property}")

        # We apply a full-move update (move_all) for speed; agents not in `agents`
        # keep their original coordinates.
        full_ids = self._agents["agent_id"].to_numpy()
        full_coords = (
            self._agents.select(self._pos_col_names)
            .to_numpy()
            .astype(np.int64, copy=False)
        )

        # Map move ids -> row indices in full_coords
        sorted_idx = np.argsort(full_ids)
        sorted_ids = full_ids[sorted_idx]
        move_ids = move_ids_srs.to_numpy()
        pos = np.searchsorted(sorted_ids, move_ids)
        if __debug__:
            if (pos >= sorted_ids.shape[0]).any() or not np.array_equal(
                sorted_ids[pos], move_ids
            ):
                raise ValueError("Some agents are not placed in the grid")
        move_row_idx = sorted_idx[pos]

        radius_per_agent: np.ndarray | None = None
        max_radius: int
        if radius_arr is not None:
            if radius_aligned_to == "placed":
                radius_per_agent = radius_arr[move_row_idx]
            else:
                radius_per_agent = radius_arr
            max_radius = int(radius_per_agent.max())
        else:
            max_radius = int(radius_scalar)

        centers = full_coords[move_row_idx]
        height = int(self._dimensions[1])
        origin_cell_id = centers[:, 0] * height + centers[:, 1]

        # Remaining capacity BEFORE movement (cells currently occupied have 0 remaining
        # capacity). Origins are not pre-freed: only agents that actually move away will
        # free their origin slot during conflict resolution.
        cap_flat = np.asarray(self.cells.capacity, dtype=np.int64).ravel(order="C")

        explain = self._explain_move_to_best_path(
            radius=radius_per_agent if radius_per_agent is not None else max_radius,
            property=property,
            include_center=include_center,
        )
        path = str(explain["path"])

        if path == "fast":
            score_flat = self.cells._property_buffer(property)
            csr = self._fastpath.neighbors_for_agents_array(
                centers=centers,
                radius=radius_per_agent if radius_per_agent is not None else max_radius,
                include_center=include_center,
            )
            # Filter candidates by capacity > 0, allowing the agent's own origin.
            if csr.cell_id.size:
                origin_rep = np.repeat(origin_cell_id, np.diff(csr.offsets))
                ok = cap_flat[csr.cell_id] > 0
                ok |= csr.cell_id == origin_rep
            else:
                ok = np.empty(0, dtype=bool)
            if ok.size and not ok.all():
                ok_int = ok.astype(np.int64, copy=False)
                counts = np.add.reduceat(ok_int, csr.offsets[:-1])
                new_offsets = np.empty_like(csr.offsets)
                new_offsets[0] = 0
                np.cumsum(counts, out=new_offsets[1:])
                csr = self._fastpath.csr(
                    offsets=new_offsets,
                    cell_id=csr.cell_id[ok],
                    radius=csr.radius[ok],
                    dim0=csr.dim0[ok],
                    dim1=csr.dim1[ok],
                )

            csr = self._fastpath.rank_candidates_array(csr, score_flat)
            dest_cell = self._fastpath.resolve_conflicts_lottery(
                rng=self.model.random,
                csr=csr,
                origin_cell_id=origin_cell_id,
                capacity_flat=cap_flat,
            )
            dest_coords = self.cells._coords_from_cell_id(dest_cell)
        else:
            # Polars neighborhood generation fallback, then reuse NumPy rank+resolve.
            neighbors_df = self.neighborhood(
                radius=radius,
                target=move_ids_srs,
                include="coords",
                include_center=include_center,
            )
            if neighbors_df.is_empty():
                dest_coords = centers
            else:
                cand_coords = (
                    neighbors_df.select(self._pos_col_names)
                    .to_numpy()
                    .astype(np.int64, copy=False)
                )
                cand_cell_id = cand_coords[:, 0] * height + cand_coords[:, 1]

                cand_radius = (
                    neighbors_df["radius"].to_numpy().astype(np.int64, copy=False)
                )
                cand_d0 = cand_coords[:, 0]
                cand_d1 = cand_coords[:, 1]

                cand_agent = neighbors_df["agent_id"].to_numpy()

                # Prefer dense row-major property buffer when available (avoids join).
                cand_score: np.ndarray
                dense_ok = (
                    (not self.cells._cells.is_empty())
                    and (self.cells._cells.height == int(np.prod(self._dimensions)))
                    and (property in self.cells._cells.columns)
                )
                if dense_ok:
                    self.cells._ensure_dense_row_major_cells()
                    score_flat = self.cells._property_buffer(property)
                    cand_score = np.asarray(score_flat[cand_cell_id], dtype=np.float64)
                else:
                    cells_df = self.cells._cells.select(
                        [*self._pos_col_names, property]
                    )
                    cand = neighbors_df.join(
                        cells_df, on=self._pos_col_names, how="left"
                    )
                    cand = cand.with_columns(
                        pl.col(property).fill_null(float("-inf")).cast(pl.Float64)
                    )
                    cand_score = cand[property].to_numpy()

                # Filter by capacity > 0, allowing origin.
                cand_agent = np.asarray(cand_agent, dtype=np.uint64)
                move_ids_np = move_ids.astype(np.uint64, copy=False)
                sort_move = np.argsort(move_ids_np, kind="stable")
                move_ids_sorted = move_ids_np[sort_move]
                origin_sorted = origin_cell_id[sort_move]
                pos_o = np.searchsorted(move_ids_sorted, cand_agent)
                if __debug__:
                    if (pos_o >= move_ids_sorted.shape[0]).any() or not np.array_equal(
                        move_ids_sorted[pos_o], cand_agent
                    ):
                        raise ValueError(
                            "Neighborhood returned agent ids not in target"
                        )
                allow_origin = cand_cell_id == origin_sorted[pos_o]
                ok = (cap_flat[cand_cell_id] > 0) | allow_origin
                pos_o = pos_o[ok]
                cand_agent = cand_agent[ok]
                cand_cell_id = cand_cell_id[ok]
                cand_radius = cand_radius[ok]
                cand_d0 = cand_d0[ok]
                cand_d1 = cand_d1[ok]
                cand_score = np.asarray(cand_score)[ok]

                # Build CSR in the same order as move_ids (vectorized).
                inv_sort_move = np.empty_like(sort_move)
                inv_sort_move[sort_move] = np.arange(sort_move.size)
                agent_idx = inv_sort_move[pos_o].astype(np.int64, copy=False)

                row_order = np.argsort(agent_idx, kind="stable")
                agent_idx_s = agent_idx[row_order]
                cand_cell_s = cand_cell_id[row_order]
                cand_rad_s = cand_radius[row_order]
                cand_d0_s = cand_d0[row_order]
                cand_d1_s = cand_d1[row_order]
                cand_score_s = cand_score[row_order]

                counts = np.bincount(
                    agent_idx_s, minlength=move_ids_np.shape[0]
                ).astype(
                    np.int64,
                    copy=False,
                )
                offsets = np.empty(move_ids_np.shape[0] + 1, dtype=np.int64)
                offsets[0] = 0
                np.cumsum(counts, out=offsets[1:])

                if offsets[-1]:
                    csr = self._fastpath.csr(
                        offsets=offsets,
                        cell_id=cand_cell_s.astype(np.int64, copy=False),
                        radius=cand_rad_s.astype(np.int64, copy=False),
                        dim0=cand_d0_s.astype(np.int64, copy=False),
                        dim1=cand_d1_s.astype(np.int64, copy=False),
                    )
                    csr = self._fastpath.rank_candidates_array_by_score(
                        csr, cand_score_s
                    )
                    dest_cell = self._fastpath.resolve_conflicts_lottery(
                        rng=self.model.random,
                        csr=csr,
                        origin_cell_id=origin_cell_id,
                        capacity_flat=cap_flat,
                    )
                    dest_coords = self.cells._coords_from_cell_id(dest_cell)
                else:
                    dest_coords = centers

        # Write destination coords back into full coord array and do a full move.
        dest_full = full_coords.copy()
        dest_full[move_row_idx] = dest_coords

        self.move_all(self._agents["agent_id"], dest_full)
        return self

    def torus_adj(self, pos: GridCoordinate | GridCoordinates) -> DataFrame:
        df_coords = self._get_df_coords(pos)
        return self._df_mod(df_coords, self._dimensions, axis="columns")

    def _calculate_differences(
        self,
        pos0: GridCoordinate | GridCoordinates | None,
        pos1: GridCoordinate | GridCoordinates | None,
        agents0: IdsLike
        | AbstractAgentSet
        | AbstractAgentSetRegistry
        | Collection[AbstractAgentSet]
        | Collection[AbstractAgentSetRegistry]
        | None,
        agents1: IdsLike
        | AbstractAgentSet
        | AbstractAgentSetRegistry
        | Collection[AbstractAgentSet]
        | Collection[AbstractAgentSetRegistry]
        | None,
    ) -> DataFrame:
        pos0_df = self._get_df_coords(pos0, agents0)
        pos1_df = self._get_df_coords(pos1, agents1)
        if __debug__ and len(pos0_df) != len(pos1_df):
            raise ValueError("objects must have the same length")
        return pos1_df - pos0_df

    def _compute_offsets(self, neighborhood_type: str) -> DataFrame:
        if neighborhood_type == "moore":
            ranges = [range(-1, 2) for _ in self._dimensions]
            directions = [d for d in product(*ranges) if any(d)]
        elif neighborhood_type == "von_neumann":
            ranges = [range(-1, 2) for _ in self._dimensions]
            directions = [
                d for d in product(*ranges) if sum(map(abs, d)) <= 1 and any(d)
            ]
        elif neighborhood_type == "hexagonal":
            if __debug__ and len(self._dimensions) > 2:
                raise ValueError(
                    "Hexagonal neighborhood is only valid for 2-dimensional grids"
                )
            directions = [
                (1, 0),
                (1, -1),
                (0, -1),
                (-1, 0),
                (-1, 1),
                (0, 1),
            ]
            in_between = [
                (-1, -1),
                (0, 1),
                (-1, 0),
                (1, 1),
                (1, 0),
                (0, -1),
            ]
            df = self._df_constructor(data=directions, columns=self._pos_col_names)
            self._in_between_offsets = self._df_with_columns(
                df,
                data=in_between,
                new_columns=["in_between_dim_0", "in_between_dim_1"],
            )
            return df
        else:
            raise ValueError("Invalid neighborhood type specified")
        return self._df_constructor(data=directions, columns=self._pos_col_names)

    def _get_df_coords(
        self,
        pos: GridCoordinate | GridCoordinates | None = None,
        agents: IdsLike
        | AbstractAgentSet
        | AbstractAgentSetRegistry
        | Collection[AbstractAgentSet]
        | Collection[AbstractAgentSetRegistry]
        | None = None,
        check_bounds: bool = True,
    ) -> DataFrame:
        agents_ids = None
        if agents is not None:
            agents_ids = self._get_ids_srs(agents)

        if __debug__:
            if pos is None and agents is None:
                raise ValueError("Neither pos or agents are specified")
            elif pos is not None and agents is not None:
                raise ValueError("Both pos and agents are specified")
            if not self.torus and pos is not None and check_bounds:
                pos = self.out_of_bounds(pos)
                if pos["out_of_bounds"].any():
                    raise ValueError(
                        "If the grid is non-toroidal, every position must be in-bound"
                    )
            if agents is not None:
                b_contained = agents_ids.is_in(self.model.sets.ids)
                if (isinstance(b_contained, Series) and not b_contained.all()) or (
                    isinstance(b_contained, bool) and not b_contained
                ):
                    raise ValueError("Some agents are not present in the model")

                b_contained = self._df_contains(self._agents, "agent_id", agents_ids)
                if (isinstance(b_contained, Series) and not b_contained.all()) or (
                    isinstance(b_contained, bool) and not b_contained
                ):
                    raise ValueError("Some agents are not placed in the grid")
                if agents_ids.n_unique() != len(agents_ids):
                    raise ValueError("Some agents are present multiple times")

        if agents_ids is not None:
            df = self._df_get_masked_df(
                self._agents, index_cols="agent_id", mask=agents_ids
            )
            df = self._df_reindex(df, agents_ids, "agent_id")
            return self._df_reset_index(df, index_cols="agent_id", drop=True)
        if isinstance(pos, DataFrame):
            return self._df_reset_index(pos[self._pos_col_names], drop=True)
        elif (
            isinstance(pos, Collection)
            and isinstance(pos[0], Collection)
            and (len(pos[0]) == len(self._dimensions))
        ):
            return self._df_constructor(
                data=pos,
                columns=self._pos_col_names,
                dtypes={col: int for col in self._pos_col_names},
            )
        elif isinstance(pos, ArrayLike) and len(pos) == len(self._dimensions):
            for i, c in enumerate(pos):
                if isinstance(c, slice):
                    start = c.start if c.start is not None else 0
                    step = c.step if c.step is not None else 1
                    stop = c.stop if c.stop is not None else self._dimensions[i]
                    pos[i] = self._srs_range(
                        name=self._pos_col_names[i],
                        start=int(start),
                        end=int(stop),
                        step=int(step),
                    )
            return self._df_constructor(
                data=[pos],
                columns=self._pos_col_names,
                dtypes={col: int for col in self._pos_col_names},
            )
        elif isinstance(pos, int) and len(self._dimensions) == 1:
            return self._df_constructor(
                data=[pos],
                columns=self._pos_col_names,
                dtypes={col: int for col in self._pos_col_names},
            )
        else:
            raise ValueError("Invalid coordinates")

    def _place_or_move_agents(
        self,
        agents: IdsLike
        | AbstractAgentSet
        | AbstractAgentSetRegistry
        | Collection[AbstractAgentSet]
        | Collection[AbstractAgentSetRegistry],
        pos: GridCoordinate | GridCoordinates,
        is_move: bool,
        *,
        trust_full_move: bool = False,
        require_full_move: bool = False,
    ) -> Grid:
        agents_input = agents
        agents = self._get_ids_srs(agents)

        # If move_all (or other trusted full-move calls) pass a Polars DataFrame of
        # coordinates, coerce it to NumPy early so the NumPy fast paths below can
        # trigger.
        if (
            is_move
            and (trust_full_move or require_full_move)
            and isinstance(pos, pl.DataFrame)
        ):
            pos_df = pos
            if "agent_id" in pos_df.columns:
                pos_df = self._df_reindex(pos_df, agents, new_index_cols="agent_id")
                pos_df = pos_df.select(self._pos_col_names)
                pos_arr = pos_df.to_numpy()
                pos = tuple(pos_arr[:, i] for i in range(pos_arr.shape[1]))
            else:
                pos_df = pos_df.select(self._pos_col_names)
                pos = pos_df.to_numpy()

        reorder_idx: np.ndarray | None = None
        if is_move and (isinstance(pos, np.ndarray) or isinstance(pos, (tuple, list))):
            input_ids: np.ndarray | None = None
            if isinstance(agents_input, Series):
                input_ids = agents_input.to_numpy()
            elif isinstance(agents_input, np.ndarray):
                input_ids = agents_input
            elif isinstance(agents_input, Sequence) and not isinstance(
                agents_input,
                (AbstractAgentSet, AbstractAgentSetRegistry),
            ):
                input_ids = np.asarray(agents_input)

            if input_ids is not None:
                input_ids = np.asarray(input_ids)
                sorted_ids = np.asarray(agents.to_numpy())
                if input_ids.shape == sorted_ids.shape:
                    perm = np.argsort(input_ids)
                    if np.array_equal(input_ids[perm], sorted_ids):
                        reorder_idx = perm

        if require_full_move:
            if self._agents.is_empty():
                raise ValueError("move_all requires agents to already be placed")
            if len(agents) != self._agents.height:
                raise ValueError("move_all requires positions for all placed agents")
            if agents.n_unique() != len(agents):
                raise ValueError("move_all requires unique agent ids")

        if (
            is_move
            and isinstance(pos, np.ndarray)
            and pos.ndim == 2
            and pos.shape[1] == len(self._pos_col_names)
            and (not self._agents.is_empty())
            and len(agents) == self._agents.height
            and agents.n_unique() == len(agents)
        ):
            membership_ok = True
            if not trust_full_move:
                trusted = (
                    os.environ.get("MESA_FRAMES_GRID_TRUST_FULL_MOVE", "")
                    .strip()
                    .lower()
                )
                trust_full_move = bool(trusted and trusted not in {"0", "false"})

            if not trust_full_move:
                membership_ok = bool(agents.is_in(self._agents["agent_id"]).all())

            if membership_ok:
                if reorder_idx is not None:
                    pos = pos[reorder_idx]
                pos_arr = pos.astype(np.int64, copy=False)
                if self._torus:
                    for i, dim in enumerate(self._dimensions):
                        pos_arr[:, i] %= int(dim)
                elif __debug__:
                    for i, dim in enumerate(self._dimensions):
                        if (pos_arr[:, i] < 0).any() or (
                            pos_arr[:, i] >= int(dim)
                        ).any():
                            raise ValueError(
                                "Some coordinates are outside the grid bounds"
                            )

                capacity = self._capacity
                if capacity and capacity != np.inf:
                    if len(self._dimensions) == 2:
                        height = int(self._dimensions[1])
                        cell_id = pos_arr[:, 0] * height + pos_arr[:, 1]
                        n_cells = int(self._dimensions[0]) * height
                    else:
                        cell_id = np.ravel_multi_index(pos_arr.T, self._dimensions)
                        n_cells = int(np.prod(self._dimensions))

                    counts = np.bincount(cell_id, minlength=n_cells)
                    if __debug__:
                        if int(counts.max()) > int(capacity):
                            raise ValueError(
                                "Not enough capacity in the space for all agents"
                            )
                    new_cap = (int(capacity) - counts).reshape(self._dimensions)
                    self.cells.capacity = new_cap

                data = {"agent_id": agents}
                for i, col in enumerate(self._pos_col_names):
                    data[col] = pos_arr[:, i]
                self._agents = self._df_constructor(
                    data=data,
                    index_cols="agent_id",
                    dtypes={"agent_id": "uint64"}
                    | {col: int for col in self._pos_col_names},
                )
                return self

        if (
            is_move
            and isinstance(pos, (tuple, list))
            and len(pos) == len(self._pos_col_names)
            and (not self._agents.is_empty())
            and len(agents) == self._agents.height
            and agents.n_unique() == len(agents)
            and all(isinstance(p, np.ndarray) and p.ndim == 1 for p in pos)
            and len(pos[0]) == len(agents)
        ):
            membership_ok = True
            if not trust_full_move:
                trusted = (
                    os.environ.get("MESA_FRAMES_GRID_TRUST_FULL_MOVE", "")
                    .strip()
                    .lower()
                )
                trust_full_move = bool(trusted and trusted not in {"0", "false"})

            if not trust_full_move:
                membership_ok = bool(agents.is_in(self._agents["agent_id"]).all())

            if membership_ok:
                if reorder_idx is not None:
                    pos = [p[reorder_idx] for p in pos]
                pos_arrs = [p.astype(np.int64, copy=False) for p in pos]
                if self._torus:
                    for i, dim in enumerate(self._dimensions):
                        pos_arrs[i] %= int(dim)
                elif __debug__:
                    for i, dim in enumerate(self._dimensions):
                        if (pos_arrs[i] < 0).any() or (pos_arrs[i] >= int(dim)).any():
                            raise ValueError(
                                "Some coordinates are outside the grid bounds"
                            )

                capacity = self._capacity
                if capacity and capacity != np.inf:
                    if len(self._dimensions) == 2:
                        height = int(self._dimensions[1])
                        cell_id = pos_arrs[0] * height + pos_arrs[1]
                        n_cells = int(self._dimensions[0]) * height
                    else:
                        cell_id = np.ravel_multi_index(
                            np.stack(pos_arrs, axis=0), self._dimensions
                        )
                        n_cells = int(np.prod(self._dimensions))
                    counts = np.bincount(cell_id, minlength=n_cells)
                    if __debug__:
                        if int(counts.max()) > int(capacity):
                            raise ValueError(
                                "Not enough capacity in the space for all agents"
                            )
                    new_cap = (int(capacity) - counts).reshape(self._dimensions)
                    self.cells.capacity = new_cap

                data = {"agent_id": agents}
                for i, col in enumerate(self._pos_col_names):
                    data[col] = pos_arrs[i]
                self._agents = self._df_constructor(
                    data=data,
                    index_cols="agent_id",
                    dtypes={"agent_id": "uint64"}
                    | {col: int for col in self._pos_col_names},
                )
                return self

        if __debug__:
            if is_move:
                if not self._df_contains(self._agents, "agent_id", agents).all():
                    warn("Some agents are not present in the grid", RuntimeWarning)
            else:
                if self._df_contains(self._agents, "agent_id", agents).any():
                    warn("Some agents are already present in the grid", RuntimeWarning)

            b_contained = agents.is_in(self.model.sets.ids)
            if (isinstance(b_contained, Series) and not b_contained.all()) or (
                isinstance(b_contained, bool) and not b_contained
            ):
                raise ValueError("Some agents are not present in the model")

            if self._capacity:
                if len(agents) > self.cells.remaining_capacity + len(
                    self._df_get_masked_df(
                        self._agents,
                        index_cols="agent_id",
                        mask=agents,
                    )
                ):
                    raise ValueError("Not enough capacity in the space for all agents")

        pos_df = self._get_df_coords(pos)
        agents_df = self._srs_to_df(agents)
        if __debug__:
            if len(agents_df) != len(pos_df):
                raise ValueError("The number of agents and positions must be equal")

        new_df = self._df_concat(
            [agents_df, pos_df], how="horizontal", index_cols="agent_id"
        )
        self.cells.update_capacity_agents(new_df, operation="movement")
        full_move = (
            is_move
            and (not self._agents.is_empty())
            and new_df.height == self._agents.height
            and agents.n_unique() == len(agents)
        )
        if full_move and not trust_full_move:
            full_move = bool(agents.is_in(self._agents["agent_id"]).all())

        if full_move:
            self._agents = new_df
        else:
            self._agents = self._df_combine_first(
                new_df, self._agents, index_cols="agent_id"
            )
        return self

    @property
    def dimensions(self) -> Sequence[int]:
        return self._dimensions

    @property
    def neighborhood_type(self) -> str:
        return self._neighborhood_type

    @property
    def torus(self) -> bool:
        return self._torus
