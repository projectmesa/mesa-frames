from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from mesa_frames import AgentSet, Grid, Model


class ExampleAgentSet(AgentSet):
    def step(self) -> None:  # pragma: no cover
        return


def _make_dense_sugar(grid: Grid, *, seed: int) -> None:
    rng = np.random.default_rng(seed)
    w, h = grid.dimensions
    coords = np.array([(i, j) for i in range(w) for j in range(h)], dtype=np.int64)
    sugar = rng.integers(0, 100, size=coords.shape[0], dtype=np.int64)
    grid.cells.set(
        coords.tolist(),
        properties={"sugar": sugar},
    )


def _place_agents_unique(grid: Grid, agent_ids: pl.Series, *, seed: int) -> None:
    rng = np.random.default_rng(seed)
    w, h = grid.dimensions
    all_cells = np.array([(i, j) for i in range(w) for j in range(h)], dtype=np.int64)
    choice = rng.choice(all_cells.shape[0], size=len(agent_ids), replace=False)
    coords = all_cells[choice]
    grid.place_agents(agent_ids, coords.tolist())


def test_move_to_best_fast_equals_df(monkeypatch: pytest.MonkeyPatch) -> None:
    model_fast = Model(seed=123)
    agents_fast = ExampleAgentSet(model_fast)
    agents_fast.add({"x": np.zeros(64, dtype=np.int64)})
    model_fast.sets.add(agents_fast)

    grid_fast = Grid(
        model_fast, dimensions=[10, 10], capacity=1, neighborhood_type="moore"
    )
    model_fast.space = grid_fast
    _make_dense_sugar(grid_fast, seed=1)

    ids = agents_fast["unique_id"]
    _place_agents_unique(grid_fast, ids, seed=2)

    # Clone state for DF run.
    model_df = Model(seed=123)
    agents_df = ExampleAgentSet(model_df)
    agents_df.add({"x": np.zeros(64, dtype=np.int64)})
    model_df.sets.add(agents_df)

    grid_df = Grid(model_df, dimensions=[10, 10], capacity=1, neighborhood_type="moore")
    model_df.space = grid_df
    _make_dense_sugar(grid_df, seed=1)
    _place_agents_unique(grid_df, agents_df["unique_id"], seed=2)

    # Force the DF path for the second run.
    monkeypatch.setenv("MESA_FRAMES_GRID_MOVE_TO_BEST_FORCE_PATH", "df")

    # Move all agents.
    grid_fast.move_to_best(ids, radius=3, property="sugar", include_center=True)
    grid_df.move_to_best(
        agents_df["unique_id"], radius=3, property="sugar", include_center=True
    )

    moved_fast = grid_fast.agents.select(["agent_id", "dim_0", "dim_1"]).sort(
        "agent_id"
    )
    moved_df = grid_df.agents.select(["agent_id", "dim_0", "dim_1"]).sort("agent_id")
    assert_frame_equal(moved_fast, moved_df, check_dtypes=False)


def test_move_to_best_per_agent_radius_fast_equals_df(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_fast = Model(seed=123)
    agents_fast = ExampleAgentSet(model_fast)
    agents_fast.add({"x": np.zeros(128, dtype=np.int64)})
    model_fast.sets.add(agents_fast)

    grid_fast = Grid(
        model_fast, dimensions=[12, 12], capacity=1, neighborhood_type="moore"
    )
    model_fast.space = grid_fast
    _make_dense_sugar(grid_fast, seed=1)

    ids = agents_fast["unique_id"]
    _place_agents_unique(grid_fast, ids, seed=2)

    # Clone state for DF run.
    model_df = Model(seed=123)
    agents_df = ExampleAgentSet(model_df)
    agents_df.add({"x": np.zeros(128, dtype=np.int64)})
    model_df.sets.add(agents_df)

    grid_df = Grid(model_df, dimensions=[12, 12], capacity=1, neighborhood_type="moore")
    model_df.space = grid_df
    _make_dense_sugar(grid_df, seed=1)
    _place_agents_unique(grid_df, agents_df["unique_id"], seed=2)

    # Per-agent radii, including 0 (stay put is allowed).
    rng = np.random.default_rng(99)
    radii = rng.integers(0, 5, size=len(ids), dtype=np.int64)

    # Force the DF path for the second run.
    monkeypatch.setenv("MESA_FRAMES_GRID_MOVE_TO_BEST_FORCE_PATH", "df")

    grid_fast.move_to_best(ids, radius=radii, property="sugar", include_center=True)
    grid_df.move_to_best(
        agents_df["unique_id"], radius=radii, property="sugar", include_center=True
    )

    moved_fast = grid_fast.agents.select(["agent_id", "dim_0", "dim_1"]).sort(
        "agent_id"
    )
    moved_df = grid_df.agents.select(["agent_id", "dim_0", "dim_1"]).sort("agent_id")
    assert_frame_equal(moved_fast, moved_df, check_dtypes=False)


def test_move_to_best_per_agent_radius_length_must_match() -> None:
    model = Model(seed=0)
    agents = ExampleAgentSet(model)
    agents.add({"x": np.zeros(8, dtype=np.int64)})
    model.sets.add(agents)

    grid = Grid(model, dimensions=[10, 10], capacity=1, neighborhood_type="moore")
    model.space = grid
    _make_dense_sugar(grid, seed=0)

    ids = agents["unique_id"]
    _place_agents_unique(grid, ids, seed=0)

    with pytest.raises(ValueError):
        grid.move_to_best(ids, radius=np.array([1, 2, 3], dtype=np.int64))


def test_move_to_best_falls_back_when_cells_not_dense() -> None:
    model = Model(seed=0)
    agents = ExampleAgentSet(model)
    agents.add({"x": np.zeros(8, dtype=np.int64)})
    model.sets.add(agents)

    grid = Grid(model, dimensions=[10, 10], capacity=1, neighborhood_type="moore")
    model.space = grid

    # Sparse cell properties (not dense)
    grid.cells.set([[0, 0]], properties={"sugar": [10]})

    ids = agents["unique_id"]
    _place_agents_unique(grid, ids, seed=1)

    explain = grid._explain_move_to_best_path(
        radius=1, property="sugar", include_center=True
    )
    assert explain["path"] == "df"

    # Should still run and keep valid coordinates.
    grid.move_to_best(ids, radius=1, property="sugar", include_center=True)
    assert grid.agents.height == len(ids)


def test_move_to_best_rejects_non_2d_grid() -> None:
    model = Model(seed=0)
    agents = ExampleAgentSet(model)
    agents.add({"x": np.zeros(2, dtype=np.int64)})
    model.sets.add(agents)

    grid = Grid(model, dimensions=[3, 3, 3], capacity=1)
    model.space = grid
    # Minimal cells property (still not enough; should fail on dims first)
    grid.cells.set([[0, 0, 0]], properties={"sugar": [1]})

    with pytest.raises(ValueError):
        grid.move_to_best(agents["unique_id"], radius=1, property="sugar")


def test_move_to_best_fast_path_selected_smoke() -> None:
    model = Model(seed=0)
    agents = ExampleAgentSet(model)
    agents.add({"x": np.zeros(10_000, dtype=np.int64)})
    model.sets.add(agents)

    grid = Grid(model, dimensions=[200, 200], capacity=1, neighborhood_type="moore")
    model.space = grid
    _make_dense_sugar(grid, seed=0)

    ids = agents["unique_id"]
    _place_agents_unique(grid, ids, seed=0)

    explain = grid._explain_move_to_best_path(
        radius=3, property="sugar", include_center=True
    )
    assert explain["path"] == "fast"
    assert "fast_kind" in explain

    # Completes without error.
    grid.move_to_best(ids, radius=3, property="sugar", include_center=True)


def test_move_to_best_explain_falls_back_when_numba_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = Model(seed=0)
    agents = ExampleAgentSet(model)
    agents.add({"x": np.zeros(64, dtype=np.int64)})
    model.sets.add(agents)

    grid = Grid(model, dimensions=[10, 10], capacity=1, neighborhood_type="moore")
    model.space = grid
    _make_dense_sugar(grid, seed=0)
    ids = agents["unique_id"]
    _place_agents_unique(grid, ids, seed=0)

    monkeypatch.setenv("MESA_FRAMES_GRID_MOVE_TO_BEST_DISABLE_NUMBA", "1")

    explain_scalar = grid._explain_move_to_best_path(
        radius=2, property="sugar", include_center=True
    )
    assert explain_scalar["path"] == "df"
    assert explain_scalar["fast_kind"] == "python"

    radii = np.full(len(ids), 2, dtype=np.int64)
    explain_per_agent = grid._explain_move_to_best_path(
        radius=radii, property="sugar", include_center=True
    )
    assert explain_per_agent["path"] == "df"
    assert explain_per_agent["fast_kind"] == "python"


def test_move_to_best_sorts_dense_cells_row_major_for_fastpath() -> None:
    model = Model(seed=0)
    agents = ExampleAgentSet(model)
    agents.add({"x": np.zeros(64, dtype=np.int64)})
    model.sets.add(agents)

    grid = Grid(model, dimensions=[10, 10], capacity=1, neighborhood_type="moore")
    model.space = grid
    _make_dense_sugar(grid, seed=0)

    ids = agents["unique_id"]
    _place_agents_unique(grid, ids, seed=0)

    # Shuffle dense cells to simulate user reordering.
    grid.cells._cells = grid.cells._cells.sample(
        n=grid.cells._cells.height, shuffle=True, seed=42
    )
    try:
        object.__delattr__(grid, "_cells_row_major_ok")
    except AttributeError:
        pass

    # Before fastpath selection, the cell_id order should be non-canonical.
    coords = grid.cells._cells.select(["dim_0", "dim_1"]).to_numpy()
    cell_id = coords[:, 0] * int(grid.dimensions[1]) + coords[:, 1]
    assert not np.array_equal(cell_id, np.arange(cell_id.size))

    explain = grid._explain_move_to_best_path(
        radius=2, property="sugar", include_center=True
    )
    assert explain["path"] == "fast"

    # After explain, cells must be sorted to row-major and cached.
    coords2 = grid.cells._cells.select(["dim_0", "dim_1"]).to_numpy()
    cell_id2 = coords2[:, 0] * int(grid.dimensions[1]) + coords2[:, 1]
    assert np.array_equal(cell_id2, np.arange(cell_id2.size))
    assert getattr(grid, "_cells_row_major_ok", False) is True

    grid.move_to_best(ids, radius=2, property="sugar", include_center=True)


def test_move_to_best_accepts_numpy_id_array() -> None:
    model = Model(seed=0)
    agents = ExampleAgentSet(model)
    agents.add({"x": np.zeros(16, dtype=np.int64)})
    model.sets.add(agents)

    grid = Grid(model, dimensions=[10, 10], capacity=1, neighborhood_type="moore")
    model.space = grid
    _make_dense_sugar(grid, seed=0)

    ids = agents["unique_id"]
    _place_agents_unique(grid, ids, seed=0)

    ids_np = ids.to_numpy().astype(np.uint64, copy=False)
    grid.move_to_best(ids_np, radius=1, property="sugar", include_center=True)


def test_move_to_best_scalar_equals_uniform_per_agent_radius(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_a = Model(seed=123)
    agents_a = ExampleAgentSet(model_a)
    agents_a.add({"x": np.zeros(64, dtype=np.int64)})
    model_a.sets.add(agents_a)

    grid_a = Grid(model_a, dimensions=[10, 10], capacity=1, neighborhood_type="moore")
    model_a.space = grid_a
    _make_dense_sugar(grid_a, seed=1)
    ids_a = agents_a["unique_id"]
    _place_agents_unique(grid_a, ids_a, seed=2)

    model_b = Model(seed=123)
    agents_b = ExampleAgentSet(model_b)
    agents_b.add({"x": np.zeros(64, dtype=np.int64)})
    model_b.sets.add(agents_b)

    grid_b = Grid(model_b, dimensions=[10, 10], capacity=1, neighborhood_type="moore")
    model_b.space = grid_b
    _make_dense_sugar(grid_b, seed=1)
    ids_b = agents_b["unique_id"]
    _place_agents_unique(grid_b, ids_b, seed=2)

    monkeypatch.setenv("MESA_FRAMES_GRID_MOVE_TO_BEST_FORCE_PATH", "fast")

    radius = 3
    radii = np.full(len(ids_a), radius, dtype=np.int64)
    grid_a.move_to_best(ids_a, radius=radius, property="sugar", include_center=True)
    grid_b.move_to_best(ids_b, radius=radii, property="sugar", include_center=True)

    moved_a = grid_a.agents.select(["agent_id", "dim_0", "dim_1"]).sort("agent_id")
    moved_b = grid_b.agents.select(["agent_id", "dim_0", "dim_1"]).sort("agent_id")
    assert_frame_equal(moved_a, moved_b, check_dtypes=False)


def test_move_to_best_per_agent_radius_torus_fast_equals_df(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_fast = Model(seed=123)
    agents_fast = ExampleAgentSet(model_fast)
    agents_fast.add({"x": np.zeros(128, dtype=np.int64)})
    model_fast.sets.add(agents_fast)

    grid_fast = Grid(
        model_fast,
        dimensions=[12, 12],
        capacity=1,
        neighborhood_type="moore",
        torus=True,
    )
    model_fast.space = grid_fast
    _make_dense_sugar(grid_fast, seed=1)

    ids = agents_fast["unique_id"]
    _place_agents_unique(grid_fast, ids, seed=2)

    model_df = Model(seed=123)
    agents_df = ExampleAgentSet(model_df)
    agents_df.add({"x": np.zeros(128, dtype=np.int64)})
    model_df.sets.add(agents_df)

    grid_df = Grid(
        model_df,
        dimensions=[12, 12],
        capacity=1,
        neighborhood_type="moore",
        torus=True,
    )
    model_df.space = grid_df
    _make_dense_sugar(grid_df, seed=1)
    _place_agents_unique(grid_df, agents_df["unique_id"], seed=2)

    rng = np.random.default_rng(99)
    radii = rng.integers(0, 10, size=len(ids), dtype=np.int64)

    monkeypatch.setenv("MESA_FRAMES_GRID_MOVE_TO_BEST_FORCE_PATH", "df")
    grid_fast.move_to_best(ids, radius=radii, property="sugar", include_center=True)
    grid_df.move_to_best(
        agents_df["unique_id"], radius=radii, property="sugar", include_center=True
    )

    moved_fast = grid_fast.agents.select(["agent_id", "dim_0", "dim_1"]).sort(
        "agent_id"
    )
    moved_df = grid_df.agents.select(["agent_id", "dim_0", "dim_1"]).sort("agent_id")
    assert_frame_equal(moved_fast, moved_df, check_dtypes=False)


def test_move_to_best_radius_series_aligned_to_placed_agents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_a = Model(seed=123)
    agents_a = ExampleAgentSet(model_a)
    agents_a.add({"x": np.zeros(64, dtype=np.int64)})
    model_a.sets.add(agents_a)

    grid_a = Grid(model_a, dimensions=[10, 10], capacity=1, neighborhood_type="moore")
    model_a.space = grid_a
    _make_dense_sugar(grid_a, seed=1)
    ids_a = agents_a["unique_id"]
    _place_agents_unique(grid_a, ids_a, seed=2)

    # Move a subset.
    move_ids = ids_a.head(16)

    # Provide radii aligned to all placed agents (grid's internal agent table order).
    full_ids = grid_a.agents["agent_id"].to_numpy()
    radius_full = np.arange(full_ids.shape[0], dtype=np.int64)
    radius_full_srs = pl.Series("radius", radius_full)

    # Compute expected per-move alignment using the same searchsorted mapping logic.
    sorted_idx = np.argsort(full_ids)
    sorted_ids = full_ids[sorted_idx]
    move_np = move_ids.to_numpy()
    pos = np.searchsorted(sorted_ids, move_np)
    move_row_idx = sorted_idx[pos]
    radius_aligned = radius_full[move_row_idx]

    # Run two equivalent fast runs.
    model_b = Model(seed=123)
    agents_b = ExampleAgentSet(model_b)
    agents_b.add({"x": np.zeros(64, dtype=np.int64)})
    model_b.sets.add(agents_b)

    grid_b = Grid(model_b, dimensions=[10, 10], capacity=1, neighborhood_type="moore")
    model_b.space = grid_b
    _make_dense_sugar(grid_b, seed=1)
    _place_agents_unique(grid_b, agents_b["unique_id"], seed=2)

    monkeypatch.setenv("MESA_FRAMES_GRID_MOVE_TO_BEST_FORCE_PATH", "fast")

    grid_a.move_to_best(move_ids, radius=radius_full_srs, property="sugar")
    grid_b.move_to_best(
        agents_b["unique_id"].head(16), radius=radius_aligned, property="sugar"
    )

    moved_a = grid_a.agents.select(["agent_id", "dim_0", "dim_1"]).sort("agent_id")
    moved_b = grid_b.agents.select(["agent_id", "dim_0", "dim_1"]).sort("agent_id")
    assert_frame_equal(moved_a, moved_b, check_dtypes=False)
