import polars as pl

from mesa_frames import Grid
from tests.space.utils import get_unique_ids


def test_move_to_available(grid_moore: Grid):
    # Test with GridCoordinate
    unique_ids = get_unique_ids(grid_moore.model)
    last = None
    different = False
    for _ in range(10):
        available_cells = grid_moore.cells.available
        space = grid_moore.move_to_available(unique_ids[0], inplace=False)
        if last is not None and not different:
            if (space.agents.select(pl.col("dim_0", "dim_1")).to_numpy() != last).any():
                different = True
        assert (
            space.agents.select(pl.col("dim_0", "dim_1")).row(0)
            in available_cells.rows()
        )
        last = space.agents.select(pl.col("dim_0", "dim_1")).to_numpy()
    assert different

    # Test with GridCoordinates
    last = None
    different = False
    for _ in range(10):
        available_cells = grid_moore.cells.available
        space = grid_moore.move_to_available(unique_ids[[0, 1]], inplace=False)
        if last is not None and not different:
            if (space.agents.select(pl.col("dim_0", "dim_1")).to_numpy() != last).any():
                different = True
        assert (
            space.agents.select(pl.col("dim_0", "dim_1")).row(0)
            in available_cells.rows()
        ) and (
            space.agents.select(pl.col("dim_0", "dim_1")).row(1)
            in available_cells.rows()
        )
        last = space.agents.select(pl.col("dim_0", "dim_1")).to_numpy()
    assert different

    # Test with AgentSet
    last = None
    different = False
    for _ in range(10):
        available_cells = grid_moore.cells.available
        space = grid_moore.move_to_available(grid_moore.model.sets, inplace=False)
        if last is not None and not different:
            if (space.agents.select(pl.col("dim_0")).to_numpy() != last).any():
                different = True
        assert (
            space.agents.select(pl.col("dim_0", "dim_1")).row(0)
            in available_cells.rows()
        ) and (
            space.agents.select(pl.col("dim_0", "dim_1")).row(1)
            in available_cells.rows()
        )
        last = space.agents.select(pl.col("dim_0")).to_numpy()
    assert different


def test_move_to_empty(grid_moore: Grid):
    # Test with GridCoordinate
    unique_ids = get_unique_ids(grid_moore.model)
    last = None
    different = False
    for _ in range(10):
        empty_cells = grid_moore.cells.empty
        space = grid_moore.move_to_empty(unique_ids[0], inplace=False)
        if last is not None and not different:
            if (space.agents.select(pl.col("dim_0", "dim_1")).to_numpy() != last).any():
                different = True
        assert (
            space.agents.filter(pl.col("agent_id") == unique_ids[0]).row(0)[1:]
            in empty_cells.rows()
        )
        last = space.agents.select(pl.col("dim_0", "dim_1")).to_numpy()
    assert different

    # Test with GridCoordinates
    last = None
    different = False
    for _ in range(10):
        empty_cells = grid_moore.cells.empty
        space = grid_moore.move_to_empty(unique_ids[[0, 1]], inplace=False)
        if last is not None and not different:
            if (space.agents.select(pl.col("dim_0", "dim_1")).to_numpy() != last).any():
                different = True
        assert (
            space.agents.select(pl.col("dim_0", "dim_1")).row(0) in empty_cells.rows()
        ) and (
            space.agents.select(pl.col("dim_0", "dim_1")).row(1) in empty_cells.rows()
        )
        last = space.agents.select(pl.col("dim_0", "dim_1")).to_numpy()
    assert different

    # Test with AgentSet
    last = None
    different = False
    for _ in range(10):
        empty_cells = grid_moore.cells.empty
        space = grid_moore.move_to_empty(grid_moore.model.sets, inplace=False)
        if last is not None and not different:
            if (space.agents.select(pl.col("dim_0")).to_numpy() != last).any():
                different = True
        assert (
            space.agents.select(pl.col("dim_0", "dim_1")).row(0) in empty_cells.rows()
        ) and (
            space.agents.select(pl.col("dim_0", "dim_1")).row(1) in empty_cells.rows()
        )
        last = space.agents.select(pl.col("dim_0")).to_numpy()
    assert different


def test_place_to_available(grid_moore: Grid):
    # Test with GridCoordinate
    unique_ids = get_unique_ids(grid_moore.model)
    last = None
    different = False
    for _ in range(10):
        available_cells = grid_moore.cells.available
        space = grid_moore.place_to_available(unique_ids[0], inplace=False)
        if last is not None and not different:
            if (space.agents.select(pl.col("dim_0", "dim_1")).to_numpy() != last).any():
                different = True
        assert (
            space.agents.select(pl.col("dim_0", "dim_1")).row(0)
            in available_cells.rows()
        )
        last = space.agents.select(pl.col("dim_0", "dim_1")).to_numpy()
    assert different

    # Test with GridCoordinates
    last = None
    different = False
    for _ in range(10):
        available_cells = grid_moore.cells.available
        space = grid_moore.place_to_available(unique_ids[[0, 1]], inplace=False)
        if last is not None and not different:
            if (space.agents.select(pl.col("dim_0", "dim_1")).to_numpy() != last).any():
                different = True
        assert (
            space.agents.select(pl.col("dim_0", "dim_1")).row(0)
            in available_cells.rows()
        ) and (
            space.agents.select(pl.col("dim_0", "dim_1")).row(1)
            in available_cells.rows()
        )
        last = space.agents.select(pl.col("dim_0", "dim_1")).to_numpy()
    assert different

    # Test with AgentSet
    last = None
    different = False
    for _ in range(10):
        available_cells = grid_moore.cells.available
        space = grid_moore.place_to_available(grid_moore.model.sets, inplace=False)
        if last is not None and not different:
            if (space.agents.select(pl.col("dim_0")).to_numpy() != last).any():
                different = True
        assert (
            space.agents.select(pl.col("dim_0", "dim_1")).row(0)
            in available_cells.rows()
        ) and (
            space.agents.select(pl.col("dim_0", "dim_1")).row(1)
            in available_cells.rows()
        )
        last = space.agents.select(pl.col("dim_0")).to_numpy()
    assert different


def test_place_to_empty(grid_moore: Grid):
    # Test with GridCoordinate
    unique_ids = get_unique_ids(grid_moore.model)
    last = None
    different = False
    for _ in range(10):
        empty_cells = grid_moore.cells.empty
        space = grid_moore.place_to_empty(unique_ids[0], inplace=False)
        if last is not None and not different:
            if (space.agents.select(pl.col("dim_0", "dim_1")).to_numpy() != last).any():
                different = True
        assert (
            space.agents.filter(pl.col("agent_id") == unique_ids[0]).row(0)[1:]
            in empty_cells.rows()
        )
        last = space.agents.select(pl.col("dim_0", "dim_1")).to_numpy()
    assert different

    # Test with GridCoordinates
    last = None
    different = False
    for _ in range(10):
        empty_cells = grid_moore.cells.empty
        space = grid_moore.place_to_empty(unique_ids[[0, 1]], inplace=False)
        if last is not None and not different:
            if (space.agents.select(pl.col("dim_0", "dim_1")).to_numpy() != last).any():
                different = True
        assert (
            space.agents.select(pl.col("dim_0", "dim_1")).row(0) in empty_cells.rows()
        ) and (
            space.agents.select(pl.col("dim_0", "dim_1")).row(1) in empty_cells.rows()
        )
        last = space.agents.select(pl.col("dim_0", "dim_1")).to_numpy()
    assert different

    # Test with AgentSet
    last = None
    different = False
    for _ in range(10):
        empty_cells = grid_moore.cells.empty
        space = grid_moore.place_to_empty(grid_moore.model.sets, inplace=False)
        if last is not None and not different:
            if (space.agents.select(pl.col("dim_0")).to_numpy() != last).any():
                different = True
        assert (
            space.agents.select(pl.col("dim_0", "dim_1")).row(0) in empty_cells.rows()
        ) and (
            space.agents.select(pl.col("dim_0", "dim_1")).row(1) in empty_cells.rows()
        )
        last = space.agents.select(pl.col("dim_0")).to_numpy()
    assert different
