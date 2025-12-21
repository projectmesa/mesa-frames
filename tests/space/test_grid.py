import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from mesa_frames import Grid, Model
from tests.space.utils import get_unique_ids
from tests.test_agentset import ExampleAgentSet, fix1_AgentSet, fix2_AgentSet


def test___init__(model: Model):
    # Test with default parameters
    grid1 = Grid(model, dimensions=[3, 3])
    assert isinstance(grid1, Grid)
    assert isinstance(grid1.agents, pl.DataFrame)
    assert grid1.agents.is_empty()
    assert isinstance(grid1.cells(), pl.DataFrame)
    assert grid1.cells().is_empty()
    assert isinstance(grid1.dimensions, list)
    assert len(grid1.dimensions) == 2
    assert isinstance(grid1.neighborhood_type, str)
    assert grid1.neighborhood_type == "moore"
    assert grid1.cells.remaining_capacity == float("inf")
    assert grid1.model == model

    # Test with capacity = 10
    grid2 = Grid(model, dimensions=[3, 3], capacity=10)
    assert grid2.cells.remaining_capacity == (10 * 3 * 3)

    # Test with torus = True
    grid3 = Grid(model, dimensions=[3, 3], torus=True)
    assert grid3.torus

    # Test with neighborhood_type = "von_neumann"
    grid4 = Grid(model, dimensions=[3, 3], neighborhood_type="von_neumann")
    assert grid4.neighborhood_type == "von_neumann"

    # Test with neighborhood_type = "moore"
    grid5 = Grid(model, dimensions=[3, 3], neighborhood_type="moore")
    assert grid5.neighborhood_type == "moore"

    # Test with neighborhood_type = "hexagonal"
    grid6 = Grid(model, dimensions=[3, 3], neighborhood_type="hexagonal")
    assert grid6.neighborhood_type == "hexagonal"


def test_get_directions(
    grid_moore: Grid,
    fix1_AgentSet: ExampleAgentSet,
    fix2_AgentSet: ExampleAgentSet,
):
    unique_ids = get_unique_ids(grid_moore.model)
    # Test with GridCoordinate
    dir = grid_moore.get_directions(pos0=[1, 1], pos1=[2, 2])
    assert isinstance(dir, pl.DataFrame)
    assert dir.select(pl.col("dim_0")).to_series().to_list() == [1]
    assert dir.select(pl.col("dim_1")).to_series().to_list() == [1]

    # Test with GridCoordinates
    dir = grid_moore.get_directions(pos0=[[0, 0], [2, 2]], pos1=[[1, 2], [1, 1]])
    assert isinstance(dir, pl.DataFrame)
    assert dir.select(pl.col("dim_0")).to_series().to_list() == [1, -1]
    assert dir.select(pl.col("dim_1")).to_series().to_list() == [2, -1]

    # Test with missing agents (raises ValueError)
    with pytest.raises(ValueError):
        grid_moore.get_directions(agents0=fix1_AgentSet, agents1=fix2_AgentSet)

    # Test with IdsLike
    grid_moore.place_agents(fix2_AgentSet, [[0, 1], [0, 2], [1, 0], [1, 2]])
    assert_frame_equal(
        grid_moore.agents,
        pl.DataFrame(
            {
                "agent_id": unique_ids[[0, 1, 4, 5, 6, 7]],
                "dim_0": [0, 1, 0, 0, 1, 1],
                "dim_1": [0, 1, 1, 2, 0, 2],
            }
        ),
        check_row_order=False,
    )

    dir = grid_moore.get_directions(
        agents0=unique_ids[[0, 1]], agents1=unique_ids[[4, 5]]
    )
    print(dir)
    assert isinstance(dir, pl.DataFrame)
    assert dir.select(pl.col("dim_0")).to_series().to_list() == [0, -1]
    assert dir.select(pl.col("dim_1")).to_series().to_list() == [1, 1]

    # Test with two AgentSets
    grid_moore.place_agents(unique_ids[[2, 3]], [[1, 1], [2, 2]])
    dir = grid_moore.get_directions(agents0=fix1_AgentSet, agents1=fix2_AgentSet)
    assert isinstance(dir, pl.DataFrame)
    assert dir.select(pl.col("dim_0")).to_series().to_list() == [0, -1, 0, -1]
    assert dir.select(pl.col("dim_1")).to_series().to_list() == [1, 1, -1, 0]

    # Test with AgentSetRegistry
    dir = grid_moore.get_directions(
        agents0=grid_moore.model.sets, agents1=grid_moore.model.sets
    )
    assert isinstance(dir, pl.DataFrame)
    assert grid_moore._df_all(dir == 0).all()

    # Test with normalize
    dir = grid_moore.get_directions(
        agents0=unique_ids[[0, 1]], agents1=unique_ids[[4, 5]], normalize=True
    )
    # Check if the vectors are normalized (length should be 1)
    assert np.allclose(
        np.sqrt(
            dir.select(pl.col("dim_0")).to_series().to_numpy() ** 2
            + dir.select(pl.col("dim_1")).to_series().to_numpy() ** 2
        ),
        1.0,
    )
    # Check specific normalized values
    assert np.allclose(
        dir.select(pl.col("dim_0")).to_series().to_list(), [0, -1 / np.sqrt(2)]
    )
    assert np.allclose(
        dir.select(pl.col("dim_1")).to_series().to_list(), [1, 1 / np.sqrt(2)]
    )


def test_get_distances(
    grid_moore: Grid,
    fix1_AgentSet: ExampleAgentSet,
    fix2_AgentSet: ExampleAgentSet,
):
    # Test with GridCoordinate
    dist = grid_moore.get_distances(pos0=[1, 1], pos1=[2, 2])
    assert isinstance(dist, pl.DataFrame)
    assert np.allclose(dist.select(pl.col("distance")).to_series().to_list(), [np.sqrt(2)])

    # Test with GridCoordinates
    dist = grid_moore.get_distances(pos0=[[0, 0], [2, 2]], pos1=[[1, 2], [1, 1]])
    assert isinstance(dist, pl.DataFrame)
    assert np.allclose(
        dist.select(pl.col("distance")).to_series().to_list(),
        [np.sqrt(5), np.sqrt(2)],
    )

    # Test with missing agents (raises ValueError)
    with pytest.raises(ValueError):
        grid_moore.get_distances(agents0=fix1_AgentSet, agents1=fix2_AgentSet)

    # Test with IdsLike
    grid_moore.place_agents(fix2_AgentSet, [[0, 1], [0, 2], [1, 0], [1, 2]])
    unique_ids = get_unique_ids(grid_moore.model)
    dist = grid_moore.get_distances(
        agents0=unique_ids[[0, 1]], agents1=unique_ids[[4, 5]]
    )
    assert isinstance(dist, pl.DataFrame)
    assert np.allclose(
        dist.select(pl.col("distance")).to_series().to_list(), [1.0, np.sqrt(2)]
    )

    # Test with two AgentSets
    grid_moore.place_agents(unique_ids[[2, 3]], [[1, 1], [2, 2]])
    dist = grid_moore.get_distances(agents0=fix1_AgentSet, agents1=fix2_AgentSet)
    assert isinstance(dist, pl.DataFrame)
    assert np.allclose(
        dist.select(pl.col("distance")).to_series().to_list(),
        [1.0, np.sqrt(2), 1.0, 1.0],
    )

    # Test with AgentSetRegistry
    dist = grid_moore.get_distances(
        agents0=grid_moore.model.sets, agents1=grid_moore.model.sets
    )
    assert grid_moore._df_all(dist == 0).all()


def test_get_neighborhood(
    grid_moore: Grid,
    grid_hexagonal: Grid,
    grid_von_neumann: Grid,
    grid_moore_torus: Grid,
):
    # Test with radius = int, pos=GridCoordinate
    neighborhood = grid_moore.get_neighborhood(radius=1, pos=[1, 1])
    assert isinstance(neighborhood, pl.DataFrame)
    assert set(neighborhood.columns) == {
        "dim_0",
        "dim_1",
        "radius",
        "dim_0_center",
        "dim_1_center",
    }
    assert neighborhood.shape == (8, 5)
    assert neighborhood.select(pl.col("dim_0")).to_series().to_list() == [
        0,
        0,
        0,
        1,
        1,
        2,
        2,
        2,
    ]
    assert neighborhood.select(pl.col("dim_1")).to_series().to_list() == [
        0,
        1,
        2,
        0,
        2,
        0,
        1,
        2,
    ]
    assert neighborhood.select(pl.col("radius")).to_series().to_list() == [1] * 8
    assert neighborhood.select(pl.col("dim_0_center")).to_series().to_list() == [1] * 8
    assert neighborhood.select(pl.col("dim_1_center")).to_series().to_list() == [1] * 8

    # Test with Sequence[int], pos=Sequence[GridCoordinate]
    neighborhood = grid_moore.get_neighborhood(radius=[1, 2], pos=[[1, 1], [2, 2]])
    neighborhood = neighborhood.sort(["dim_0_center", "dim_1_center", "radius"])
    assert isinstance(neighborhood, pl.DataFrame)
    assert neighborhood.shape == (8 + 6, 5)
    assert (
        neighborhood.select(pl.col("radius")).to_series().to_list() == [1] * 11 + [2] * 3
    )
    assert (
        neighborhood.select(pl.col("dim_0_center")).to_series().to_list() == [1] * 8 + [2] * 6
    )
    assert (
        neighborhood.select(pl.col("dim_1_center")).to_series().to_list() == [1] * 8 + [2] * 6
    )
    neighborhood = neighborhood.sort(["dim_0", "dim_1"])
    assert (
        neighborhood.select(pl.col("dim_0")).to_series().to_list()
        == [0] * 5 + [1] * 4 + [2] * 5
    )
    assert neighborhood.select(pl.col("dim_1")).to_series().to_list() == [
        0,
        0,
        1,
        2,
        2,
        0,
        1,
        2,
        2,
        0,
        0,
        1,
        1,
        2,
    ]
    unique_ids = get_unique_ids(grid_moore.model)
    grid_moore.place_agents(unique_ids[[0, 1]], [[1, 1], [2, 2]])

    # Test with agent=int, pos=GridCoordinate
    neighborhood = grid_moore.get_neighborhood(radius=1, agents=unique_ids[0])
    neighborhood = neighborhood.sort(["dim_0_center", "dim_1_center", "radius"])
    assert isinstance(neighborhood, pl.DataFrame)
    assert neighborhood.shape == (8, 5)
    assert neighborhood.select(pl.col("dim_0")).to_series().to_list() == [
        0,
        0,
        0,
        1,
        1,
        2,
        2,
        2,
    ]
    assert neighborhood.select(pl.col("dim_1")).to_series().to_list() == [
        0,
        1,
        2,
        0,
        2,
        0,
        1,
        2,
    ]
    assert neighborhood.select(pl.col("radius")).to_series().to_list() == [1] * 8
    assert neighborhood.select(pl.col("dim_0_center")).to_series().to_list() == [1] * 8
    assert neighborhood.select(pl.col("dim_1_center")).to_series().to_list() == [1] * 8

    # Test with agent=Sequence[int], pos=Sequence[GridCoordinate]
    neighborhood = grid_moore.get_neighborhood(radius=[1, 2], agents=unique_ids[[0, 1]])
    neighborhood = neighborhood.sort(["dim_0_center", "dim_1_center", "radius"])
    assert isinstance(neighborhood, pl.DataFrame)
    assert neighborhood.shape == (8 + 6, 5)
    assert (
        neighborhood.select(pl.col("radius")).to_series().to_list() == [1] * 11 + [2] * 3
    )
    assert (
        neighborhood.select(pl.col("dim_0_center")).to_series().to_list() == [1] * 8 + [2] * 6
    )
    assert (
        neighborhood.select(pl.col("dim_1_center")).to_series().to_list() == [1] * 8 + [2] * 6
    )
    neighborhood = neighborhood.sort(["dim_0", "dim_1"])
    assert (
        neighborhood.select(pl.col("dim_0")).to_series().to_list()
        == [0] * 5 + [1] * 4 + [2] * 5
    )
    assert neighborhood.select(pl.col("dim_1")).to_series().to_list() == [
        0,
        0,
        1,
        2,
        2,
        0,
        1,
        2,
        2,
        0,
        0,
        1,
        1,
        2,
    ]

    # Test with include_center
    neighborhood = grid_moore.get_neighborhood(radius=1, pos=[1, 1], include_center=True)
    neighborhood = neighborhood.sort(["dim_0_center", "dim_1_center", "radius"])
    assert isinstance(neighborhood, pl.DataFrame)
    assert neighborhood.shape == (9, 5)
    assert neighborhood.select(pl.col("dim_0")).to_series().to_list() == [
        1,
        0,
        0,
        0,
        1,
        1,
        2,
        2,
        2,
    ]
    assert neighborhood.select(pl.col("dim_1")).to_series().to_list() == [
        1,
        0,
        1,
        2,
        0,
        2,
        0,
        1,
        2,
    ]
    assert neighborhood.select(pl.col("radius")).to_series().to_list() == [0] + [1] * 8
    assert neighborhood.select(pl.col("dim_0_center")).to_series().to_list() == [1] * 9
    assert neighborhood.select(pl.col("dim_1_center")).to_series().to_list() == [1] * 9

    # Test with torus
    neighborhood = grid_moore_torus.get_neighborhood(radius=1, pos=[0, 0])
    neighborhood = neighborhood.sort(["dim_0_center", "dim_1_center", "radius"])
    assert isinstance(neighborhood, pl.DataFrame)
    assert neighborhood.shape == (8, 5)
    assert neighborhood.select(pl.col("dim_0")).to_series().to_list() == [
        2,
        2,
        2,
        0,
        0,
        1,
        1,
        1,
    ]
    assert neighborhood.select(pl.col("dim_1")).to_series().to_list() == [
        2,
        0,
        1,
        2,
        1,
        2,
        0,
        1,
    ]
    assert neighborhood.select(pl.col("radius")).to_series().to_list() == [1] * 8
    assert neighborhood.select(pl.col("dim_0_center")).to_series().to_list() == [0] * 8
    assert neighborhood.select(pl.col("dim_1_center")).to_series().to_list() == [0] * 8

    # Test with radius and pos of different length
    with pytest.raises(ValueError):
        grid_moore.get_neighborhood(radius=[1, 2], pos=[1, 1])

    # Test with von_neumann neighborhood
    neighborhood = grid_von_neumann.get_neighborhood(radius=1, pos=[1, 1])
    assert isinstance(neighborhood, pl.DataFrame)
    assert neighborhood.shape == (4, 5)
    assert neighborhood.select(pl.col("dim_0")).to_series().to_list() == [
        0,
        1,
        1,
        2,
    ]
    assert neighborhood.select(pl.col("dim_1")).to_series().to_list() == [
        1,
        0,
        2,
        1,
    ]
    assert neighborhood.select(pl.col("radius")).to_series().to_list() == [1] * 4
    assert neighborhood.select(pl.col("dim_0_center")).to_series().to_list() == [1] * 4
    assert neighborhood.select(pl.col("dim_1_center")).to_series().to_list() == [1] * 4

    # Test with hexagonal neighborhood (odd cell [2,1] and even cell [2,2])
    neighborhood = grid_hexagonal.get_neighborhood(radius=[2, 3], pos=[[5, 4], [5, 5]])
    assert isinstance(neighborhood, pl.DataFrame)
    assert neighborhood.shape == (6 * 2 + 12 * 2 + 18, 5)

    # Sort the neighborhood for consistent ordering
    neighborhood = neighborhood.sort(
        ["dim_0_center", "dim_1_center", "radius", "dim_0", "dim_1"]
    )

    # Expected neighbors for [5,4] and [5,5]
    expected_neighbors = [
        # Neighbors of [5,4]
        # radius 1
        (4, 4),
        (4, 5),
        (5, 3),
        (5, 5),
        (6, 3),
        (6, 4),
        # radius 2
        (3, 4),
        (3, 6),
        (4, 2),
        (4, 5),
        (4, 6),
        (5, 2),
        (5, 5),
        (5, 6),
        (6, 3),
        (7, 2),
        (7, 3),
        (7, 4),
        # Neighbors of [5,5]
        # radius 1
        (4, 5),
        (4, 6),
        (5, 4),
        (5, 6),
        (6, 4),
        (6, 5),
        # radius 2
        (3, 5),
        (3, 7),
        (4, 3),
        (4, 6),
        (4, 7),
        (5, 3),
        (5, 6),
        (5, 7),
        (6, 4),
        (7, 3),
        (7, 4),
        (7, 5),
        # radius 3
        (2, 5),
        (2, 8),
        (3, 2),
        (3, 6),
        (3, 8),
        (4, 2),
        (4, 7),
        (4, 8),
        (5, 2),
        (5, 6),
        (5, 7),
        (5, 8),
        (6, 3),
        (7, 4),
        (8, 2),
        (8, 3),
        (8, 4),
        (8, 5),
    ]

    assert (
        list(
            zip(
                neighborhood.select(pl.col("dim_0")).to_series().to_list(),
                neighborhood.select(pl.col("dim_1")).to_series().to_list(),
            )
        )
        == expected_neighbors
    )


def test_get_neighbors(
    fix2_AgentSet: ExampleAgentSet,
    grid_moore: Grid,
    grid_hexagonal: Grid,
    grid_von_neumann: Grid,
    grid_moore_torus: Grid,
):
    # Place agents in the grid
    unique_ids = get_unique_ids(grid_moore.model)
    grid_moore.move_agents(
        unique_ids[[0, 1, 2, 3, 4, 5, 6, 7]],
        [[0, 0], [0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1], [2, 2]],
    )

    # Test with radius = int, pos=GridCoordinate
    neighbors = grid_moore.get_neighbors(radius=1, pos=[1, 1])
    assert_frame_equal(
        neighbors,
        pl.DataFrame(
            {
                "agent_id": unique_ids[[0, 1, 2, 3, 4, 5, 6, 7]],
                "dim_0": [0, 0, 0, 1, 1, 2, 2, 2],
                "dim_1": [0, 1, 2, 0, 2, 0, 1, 2],
            }
        ),
        check_row_order=False,
    )

    # Test with Sequence[int], pos=Sequence[GridCoordinate]
    neighbors = grid_moore.get_neighbors(radius=[1, 2], pos=[[1, 1], [2, 2]])
    assert_frame_equal(
        neighbors,
        pl.DataFrame(
            {
                "agent_id": unique_ids[[0, 1, 2, 3, 4, 5, 6, 7]],
                "dim_0": [0, 0, 0, 1, 1, 2, 2, 2],
                "dim_1": [0, 1, 2, 0, 2, 0, 1, 2],
            }
        ),
        check_row_order=False,
    )

    # Test with agent=int
    neighbors = grid_moore.get_neighbors(radius=1, agents=unique_ids[0])
    assert_frame_equal(
        neighbors,
        pl.DataFrame({"agent_id": unique_ids[[1, 3]], "dim_0": [0, 1], "dim_1": [1, 0]}),
        check_row_order=False,
    )

    # Test with agent=Sequence[int]
    neighbors = grid_moore.get_neighbors(radius=[1, 2], agents=unique_ids[[0, 7]])
    assert_frame_equal(
        neighbors,
        pl.DataFrame(
            {
                "agent_id": unique_ids[[0, 1, 2, 3, 4, 5, 6]],
                "dim_0": [0, 0, 0, 1, 1, 2, 2],
                "dim_1": [0, 1, 2, 0, 2, 0, 1],
            }
        ),
        check_row_order=False,
    )

    # Test with include_center
    neighbors = grid_moore.get_neighbors(radius=1, pos=[1, 1], include_center=True)
    assert_frame_equal(
        neighbors,
        pl.DataFrame(
            {
                "agent_id": unique_ids[[0, 1, 2, 3, 4, 5, 6, 7]],
                "dim_0": [0, 0, 0, 1, 1, 2, 2, 2],
                "dim_1": [0, 1, 2, 0, 2, 0, 1, 2],
            }
        ),
        check_row_order=False,
    )

    # Test with torus
    grid_moore_torus.move_agents(
        unique_ids[[0, 1, 2, 3, 4, 5, 6, 7]],
        [[2, 2], [2, 0], [2, 1], [0, 2], [0, 1], [1, 2], [1, 0], [1, 1]],
    )
    neighbors = grid_moore_torus.get_neighbors(radius=1, pos=[0, 0])
    assert_frame_equal(
        neighbors,
        pl.DataFrame(
            {
                "agent_id": unique_ids[[0, 1, 2, 3, 4, 5, 6, 7]],
                "dim_0": [2, 2, 2, 0, 0, 1, 1, 1],
                "dim_1": [2, 0, 1, 2, 1, 2, 0, 1],
            }
        ),
        check_row_order=False,
    )

    # Test with radius and pos of different length
    with pytest.raises(ValueError):
        grid_moore.get_neighbors(radius=[1, 2], pos=[1, 1])

    # Test with von_neumann neighborhood
    grid_von_neumann.move_agents(
        unique_ids[[0, 1, 2, 3]], [[0, 1], [1, 0], [1, 2], [2, 1]]
    )
    neighbors = grid_von_neumann.get_neighbors(radius=1, pos=[1, 1])
    assert_frame_equal(
        neighbors,
        pl.DataFrame(
            {
                "agent_id": unique_ids[[0, 1, 2, 3]],
                "dim_0": [0, 1, 1, 2],
                "dim_1": [1, 0, 2, 1],
            }
        ),
        check_row_order=False,
    )

    # Test with hexagonal neighborhood (odd cell [5,4] and even cell [5,5])
    unique_ids = get_unique_ids(grid_hexagonal.model)
    grid_hexagonal.move_agents(
        unique_ids[:8],
        [[4, 4], [4, 5], [5, 3], [5, 5], [6, 3], [6, 4], [5, 4], [5, 6]],
    )
    neighbors = grid_hexagonal.get_neighbors(radius=[2, 3], pos=[[5, 4], [5, 5]])
    assert_frame_equal(
        neighbors,
        pl.DataFrame(
            {
                "agent_id": unique_ids[:8],
                "dim_0": [4, 4, 5, 5, 6, 6, 5, 5],
                "dim_1": [4, 5, 3, 5, 3, 4, 4, 6],
            }
        ),
        check_row_order=False,
        check_column_order=False,
    )


def test_out_of_bounds(grid_moore: Grid):
    # Test with GridCoordinate
    out_of_bounds = grid_moore.out_of_bounds([11, 11])
    assert isinstance(out_of_bounds, pl.DataFrame)
    assert out_of_bounds.shape == (1, 3)
    assert out_of_bounds.columns == ["dim_0", "dim_1", "out_of_bounds"]
    assert out_of_bounds.row(0) == (11, 11, True)

    # Test with GridCoordinates
    out_of_bounds = grid_moore.out_of_bounds([[0, 0], [11, 11]])
    assert isinstance(out_of_bounds, pl.DataFrame)
    assert out_of_bounds.shape == (2, 3)
    assert out_of_bounds.columns == ["dim_0", "dim_1", "out_of_bounds"]
    assert out_of_bounds.row(0) == (0, 0, False)
    assert out_of_bounds.row(1) == (11, 11, True)


def test_torus_adj(grid_moore: Grid, grid_moore_torus: Grid):
    # Test with non-toroidal grid
    with pytest.raises(ValueError):
        grid_moore.torus_adj([10, 10])

    # Test with toroidal grid (GridCoordinate)
    adj_df = grid_moore_torus.torus_adj([10, 8])
    assert isinstance(adj_df, pl.DataFrame)
    assert adj_df.shape == (1, 2)
    assert adj_df.columns == ["dim_0", "dim_1"]
    assert adj_df.row(0) == (1, 2)

    # Test with toroidal grid (GridCoordinates)
    adj_df = grid_moore_torus.torus_adj([[10, 8], [15, 11]])
    assert isinstance(adj_df, pl.DataFrame)
    assert adj_df.shape == (2, 2)
    assert adj_df.columns == ["dim_0", "dim_1"]
    assert adj_df.row(0) == (1, 2)
    assert adj_df.row(1) == (0, 2)


def test_dimensions(grid_moore: Grid):
    assert isinstance(grid_moore.dimensions, list)
    assert len(grid_moore.dimensions) == 2


def test_neighborhood_type(
    grid_moore: Grid,
    grid_von_neumann: Grid,
    grid_hexagonal: Grid,
):
    assert grid_moore.neighborhood_type == "moore"
    assert grid_von_neumann.neighborhood_type == "von_neumann"
    assert grid_hexagonal.neighborhood_type == "hexagonal"


def test_torus(model: Model, grid_moore: Grid):
    assert not grid_moore.torus

    grid_2 = Grid(model, [3, 3], torus=True)
    assert grid_2.torus
