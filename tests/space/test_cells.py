import polars as pl
import pytest
from polars.testing import assert_frame_equal

from mesa_frames import Grid, Model
from tests.space.utils import get_unique_ids


def test_get_cells(grid_moore: Grid):
    # Test with None (all cells)
    result = grid_moore.cells()
    assert isinstance(result, pl.DataFrame)
    assert result.select(pl.col("dim_0")).to_series().to_list() == [0, 1]
    assert result.select(pl.col("dim_1")).to_series().to_list() == [0, 1]
    assert result.select(pl.col("capacity")).to_series().to_list() == [1, 3]
    assert result.select(pl.col("property_0")).to_series().to_list() == [
        "value_0",
        "value_0",
    ]

    # Test with GridCoordinate
    single = grid_moore.cells([0, 0])
    assert isinstance(single, pl.DataFrame)
    assert single.height == 1
    assert single.select(pl.col("dim_0")).to_series().to_list() == [0]
    assert single.select(pl.col("dim_1")).to_series().to_list() == [0]
    assert single.select(pl.col("capacity")).to_series().to_list() == [1]
    assert single.select(pl.col("property_0")).to_series().to_list() == ["value_0"]

    # Test with GridCoordinates (order should match input)
    multi = grid_moore.cells([[1, 1], [0, 0]])
    assert isinstance(multi, pl.DataFrame)
    assert multi.height == 2
    coords = list(zip(multi["dim_0"].to_list(), multi["dim_1"].to_list()))
    assert set(coords) == {(0, 0), (1, 1)}
    expected = pl.DataFrame({"dim_0": [0, 1], "dim_1": [0, 1], "capacity": [1, 3]})
    assert_frame_equal(
        multi.select(["dim_0", "dim_1", "capacity"]).sort(["dim_0", "dim_1"]),
        expected.sort(["dim_0", "dim_1"]),
        check_dtypes=False,
    )


def test_cells_coords_accessor(grid_moore: Grid):
    coords = grid_moore.cells.coords
    assert isinstance(coords, pl.DataFrame)
    assert coords.columns == ["dim_0", "dim_1"]
    assert_frame_equal(
        coords,
        grid_moore.cells(include="properties").select(["dim_0", "dim_1"]),
        check_dtypes=False,
        check_row_order=False,
    )


def test_set_cells_dense_grid_delta_update(model: Model):
    grid = Grid(model, dimensions=[3, 3], capacity=2)
    # Initialize a dense cells table.
    dim0 = pl.Series("dim_0", pl.arange(3, eager=True)).to_frame()
    dim1 = pl.Series("dim_1", pl.arange(3, eager=True)).to_frame()
    all_coords = dim0.join(dim1, how="cross")
    grid.cells.set(
        all_coords,
        properties={
            "capacity": pl.repeat(2, all_coords.height, eager=True).cast(pl.Int64),
            "sugar": pl.repeat(1, all_coords.height, eager=True).cast(pl.Int64),
        },
    )
    # Small delta update: only touch one cell.
    grid.cells.set([[1, 2]], properties={"sugar": 9})
    cell = grid.cells([1, 2], include="properties")
    assert int(cell["sugar"][0]) == 9
    # Unaffected cells retain original values.
    other = grid.cells([0, 0], include="properties")
    assert int(other["sugar"][0]) == 1


def test_is_available(grid_moore: Grid):
    # Test with GridCoordinate
    result = grid_moore.cells.is_available([0, 0])
    assert isinstance(result, pl.DataFrame)
    assert result.select(pl.col("available")).to_series().to_list() == [False]
    result = grid_moore.cells.is_available([1, 1])
    assert result.select(pl.col("available")).to_series().to_list() == [True]

    # Test with GridCoordinates
    result = grid_moore.cells.is_available([[0, 0], [1, 1]])
    assert result.select(pl.col("available")).to_series().to_list() == [False, True]


def test_is_empty(grid_moore: Grid):
    # Test with GridCoordinate
    result = grid_moore.cells.is_empty([0, 0])
    assert isinstance(result, pl.DataFrame)
    assert result.select(pl.col("empty")).to_series().to_list() == [False]
    result = grid_moore.cells.is_empty([1, 1])
    assert result.select(pl.col("empty")).to_series().to_list() == [False]

    # Test with GridCoordinates
    result = grid_moore.cells.is_empty([[0, 0], [1, 1]])
    assert result.select(pl.col("empty")).to_series().to_list() == [False, False]


def test_is_full(grid_moore: Grid):
    # Test with GridCoordinate
    result = grid_moore.cells.is_full([0, 0])
    assert isinstance(result, pl.DataFrame)
    assert result.select(pl.col("full")).to_series().to_list() == [True]
    result = grid_moore.cells.is_full([1, 1])
    assert result.select(pl.col("full")).to_series().to_list() == [False]

    # Test with GridCoordinates
    result = grid_moore.cells.is_full([[0, 0], [1, 1]])
    assert result.select(pl.col("full")).to_series().to_list() == [True, False]


def test_available_cells(grid_moore: Grid):
    result = grid_moore.cells.available
    assert len(result) == 8
    assert isinstance(result, pl.DataFrame)
    assert result.columns == ["dim_0", "dim_1"]


def test_empty_cells(grid_moore: Grid):
    result = grid_moore.cells.empty
    assert len(result) == 7
    assert isinstance(result, pl.DataFrame)
    assert result.columns == ["dim_0", "dim_1"]


def test_full_cells(grid_moore: Grid):
    grid_moore.cells.set([[0, 0], [1, 1]], {"capacity": 1})
    result = grid_moore.cells.full
    assert len(result) == 2
    assert isinstance(result, pl.DataFrame)
    assert result.columns == ["dim_0", "dim_1"]
    assert (
        (
            result.select(pl.col("dim_0") == 0).to_series()
            & (result.select(pl.col("dim_1") == 0)).to_series()
        )
        | (
            result.select(pl.col("dim_0") == 1).to_series()
            & (result.select(pl.col("dim_1") == 1)).to_series()
        )
    ).all()


def test_cells(grid_moore: Grid):
    result = grid_moore.cells()
    unique_ids = get_unique_ids(grid_moore.model)
    assert_frame_equal(
        result,
        pl.DataFrame(
            {
                "dim_0": [0, 1],
                "dim_1": [0, 1],
                "capacity": [1, 3],
                "property_0": ["value_0", "value_0"],
                "agent_id": unique_ids[[0, 1]],
            }
        ),
        check_dtypes=False,
    )


def test_remaining_capacity(grid_moore: Grid):
    assert grid_moore.cells.remaining_capacity == (3 * 3 * 2 - 2)


def test_random_pos(grid_moore: Grid):
    different = False
    last = None
    for _ in range(10):
        random_pos = grid_moore.cells.sample(5)
        assert isinstance(random_pos, pl.DataFrame)
        assert len(random_pos) == 5
        assert random_pos.columns == ["dim_0", "dim_1"]
        assert (
            not grid_moore.out_of_bounds(random_pos)
            .select(pl.col("out_of_bounds"))
            .to_series()
            .any()
        )
        if last is not None and not different:
            if (last.to_numpy() != random_pos.to_numpy()).any():
                different = True
            break
        last = random_pos
    assert different


def test_sample_cells(grid_moore: Grid):
    # Test with default parameters
    replacement = False
    same = True
    last = None
    for _ in range(10):
        result = grid_moore.cells.sample(10)
        assert len(result) == 10
        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["dim_0", "dim_1"]
        counts = result.group_by("dim_0", "dim_1").agg(pl.len())
        assert (counts.select(pl.col("len")) <= 2).to_series().all()
        if not replacement and (counts.select(pl.col("len")) > 1).to_series().any():
            replacement = True
        if same and last is not None:
            same = (result.to_numpy() == last).all()
        if not same and replacement:
            break
        last = result.to_numpy()
    assert replacement and not same

    # Test with too many samples
    with pytest.raises(AssertionError):
        grid_moore.cells.sample(100)

    # Test with 'empty' cell_type
    result = grid_moore.cells.sample(14, cell_type="empty")
    assert len(result) == 14
    assert isinstance(result, pl.DataFrame)
    assert result.columns == ["dim_0", "dim_1"]
    counts = result.group_by("dim_0", "dim_1").agg(pl.len())

    ## (0, 1) and (1, 1) are not in the result
    assert not (
        result.select(pl.col("dim_0") == 0).to_series()
        & (result.select(pl.col("dim_1") == 0)).to_series()
    ).any(), "Found (0, 1) in the result"
    assert not (
        result.select(pl.col("dim_0") == 1).to_series()
        & (result.select(pl.col("dim_1") == 1)).to_series()
    ).any(), "Found (1, 1) in the result"

    # 14 should be the max number of empty cells
    with pytest.raises(AssertionError):
        grid_moore.cells.sample(15, cell_type="empty")

    # Test with 'available' cell_type
    result = grid_moore.cells.sample(16, cell_type="available")
    assert len(result) == 16
    assert isinstance(result, pl.DataFrame)
    assert result.columns == ["dim_0", "dim_1"]
    counts = result.group_by("dim_0", "dim_1").agg(pl.len())

    # 16 should be the max number of available cells
    with pytest.raises(AssertionError):
        grid_moore.cells.sample(17, cell_type="available")

    # Test with 'full' cell_type and no replacement
    grid_moore.cells.set([[0, 0], [1, 1]], properties={"capacity": 1})
    result = grid_moore.cells.sample(2, cell_type="full", with_replacement=False)
    assert len(result) == 2
    assert isinstance(result, pl.DataFrame)
    assert result.columns == ["dim_0", "dim_1"]
    assert (
        (
            result.select(pl.col("dim_0") == 0).to_series()
            & (result.select(pl.col("dim_1") == 0)).to_series()
        )
        | (
            result.select(pl.col("dim_0") == 1).to_series()
            & (result.select(pl.col("dim_1") == 1)).to_series()
        )
    ).all()
    # 2 should be the max number of full cells
    with pytest.raises(AssertionError):
        grid_moore.cells.sample(3, cell_type="full", with_replacement=False)


def test_set_cells(model: Model):
    # Initialize Grid
    grid_moore = Grid(model, dimensions=[3, 3], capacity=2)

    # Test with GridCoordinate
    grid_moore.cells.set([0, 0], properties={"capacity": 1, "property_0": "value_0"})
    assert grid_moore.cells.remaining_capacity == (2 * 3 * 3 - 1)
    cell_df = grid_moore.cells([0, 0])
    assert cell_df["capacity"][0] == 1
    assert cell_df["property_0"][0] == "value_0"

    # Test with GridCoordinates
    grid_moore.cells.set(
        [[1, 1], [2, 2]], properties={"capacity": 3, "property_1": "value_1"}
    )
    assert grid_moore.cells.remaining_capacity == (2 * 3 * 3 - 1 + 2)
    cell_df = grid_moore.cells([[1, 1], [2, 2]])
    assert cell_df["capacity"][0] == 3
    assert cell_df["property_1"][0] == "value_1"
    assert cell_df["capacity"][1] == 3
    assert cell_df["property_1"][1] == "value_1"

    cell_df = grid_moore.cells([0, 0])
    assert cell_df["capacity"][0] == 1
    assert cell_df["property_0"][0] == "value_0"

    # Test with DataFrame
    df = pl.DataFrame({"dim_0": [0, 1, 2], "dim_1": [0, 1, 2], "capacity": [2, 2, 2]})
    grid_moore.cells.set(df)
    assert grid_moore.cells.remaining_capacity == (2 * 3 * 3)

    cells_df = grid_moore.cells([[0, 0], [1, 1], [2, 2]])
    assert cells_df["capacity"][0] == 2
    assert cells_df["capacity"][1] == 2
    assert cells_df["capacity"][2] == 2
    assert cells_df["property_0"][0] == "value_0"
    assert cells_df["property_1"][1] == "value_1"
    assert cells_df["property_1"][2] == "value_1"

    # Add 2 agents to a cell, then set the cell capacity to 1
    unique_ids = get_unique_ids(grid_moore.model)
    grid_moore.place_agents(unique_ids[[1, 2]], [[0, 0], [0, 0]])
    with pytest.raises(AssertionError):
        grid_moore.cells.set([0, 0], properties={"capacity": 1})
