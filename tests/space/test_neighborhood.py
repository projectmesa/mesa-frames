import polars as pl
import pytest
from polars.testing import assert_frame_equal

from mesa_frames import Grid
from tests.space.utils import get_unique_ids
from tests.test_agentset import ExampleAgentSet


def test_get_neighborhood(
    grid_moore: Grid,
    grid_hexagonal: Grid,
    grid_von_neumann: Grid,
    grid_moore_torus: Grid,
):
    # Test with radius = int, pos=GridCoordinate
    neighborhood = grid_moore.neighborhood(radius=1, target=[1, 1])
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
    neighborhood = grid_moore.neighborhood(radius=[1, 2], target=[[1, 1], [2, 2]])
    neighborhood = neighborhood.sort(["dim_0_center", "dim_1_center", "radius"])
    assert isinstance(neighborhood, pl.DataFrame)
    assert neighborhood.shape == (8 + 6, 5)
    assert (
        neighborhood.select(pl.col("radius")).to_series().to_list()
        == [1] * 11 + [2] * 3
    )
    assert (
        neighborhood.select(pl.col("dim_0_center")).to_series().to_list()
        == [1] * 8 + [2] * 6
    )
    assert (
        neighborhood.select(pl.col("dim_1_center")).to_series().to_list()
        == [1] * 8 + [2] * 6
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
    neighborhood = grid_moore.neighborhood(radius=1, target=unique_ids[0])
    neighborhood = neighborhood.sort(["dim_0_center", "dim_1_center", "radius"])
    assert isinstance(neighborhood, pl.DataFrame)
    assert neighborhood.shape == (8, 6)
    assert (
        neighborhood.select(pl.col("agent_id")).to_series().to_list()
        == [unique_ids[0]] * 8
    )
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
    neighborhood = grid_moore.neighborhood(radius=[1, 2], target=unique_ids[[0, 1]])
    neighborhood = neighborhood.sort(["dim_0_center", "dim_1_center", "radius"])
    assert isinstance(neighborhood, pl.DataFrame)
    assert neighborhood.shape == (8 + 6, 6)
    assert (
        neighborhood.select(pl.col("agent_id")).to_series().to_list()
        == [unique_ids[0]] * 8 + [unique_ids[1]] * 6
    )
    assert (
        neighborhood.select(pl.col("radius")).to_series().to_list()
        == [1] * 11 + [2] * 3
    )
    assert (
        neighborhood.select(pl.col("dim_0_center")).to_series().to_list()
        == [1] * 8 + [2] * 6
    )
    assert (
        neighborhood.select(pl.col("dim_1_center")).to_series().to_list()
        == [1] * 8 + [2] * 6
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
    neighborhood = grid_moore.neighborhood(radius=1, target=[1, 1], include_center=True)
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
    neighborhood = grid_moore_torus.neighborhood(radius=1, target=[0, 0])
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
        grid_moore.neighborhood(radius=[1, 2], target=[1, 1])

    # Test with von_neumann neighborhood
    neighborhood = grid_von_neumann.neighborhood(radius=1, target=[1, 1])
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
    neighborhood = grid_hexagonal.neighborhood(radius=[2, 3], target=[[5, 4], [5, 5]])
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
    neighbors = grid_moore.neighborhood(radius=1, target=[1, 1], include="agents")
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
    neighbors = grid_moore.neighborhood(
        radius=[1, 2], target=[[1, 1], [2, 2]], include="agents"
    )
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
    neighbors = grid_moore.neighborhood(
        radius=1, target=unique_ids[0], include="agents"
    )
    assert_frame_equal(
        neighbors,
        pl.DataFrame({"agent_id": unique_ids[[1, 3]], "dim_0": [0, 1], "dim_1": [1, 0]}),
        check_row_order=False,
    )

    # Test with agent=Sequence[int]
    neighbors = grid_moore.neighborhood(
        radius=[1, 2], target=unique_ids[[0, 7]], include="agents"
    )
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
    neighbors = grid_moore.neighborhood(
        radius=1, target=[1, 1], include="agents", include_center=True
    )
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
    neighbors = grid_moore_torus.neighborhood(
        radius=1, target=[0, 0], include="agents"
    )
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
        grid_moore.neighborhood(radius=[1, 2], target=[1, 1])

    # Test with von_neumann neighborhood
    grid_von_neumann.move_agents(
        unique_ids[[0, 1, 2, 3]], [[0, 1], [1, 0], [1, 2], [2, 1]]
    )
    neighbors = grid_von_neumann.neighborhood(
        radius=1, target=[1, 1], include="agents"
    )
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
    neighbors = grid_hexagonal.neighborhood(
        radius=[2, 3], target=[[5, 4], [5, 5]], include="agents"
    )
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
