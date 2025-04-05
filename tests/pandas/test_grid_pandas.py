import numpy as np
import pandas as pd
import polars as pl
import pytest
import typeguard as tg

from mesa_frames import GridPandas, ModelDF
from mesa_frames.abstract.agents import AgentSetDF
from tests.pandas.test_agentset_pandas import (
    ExampleAgentSetPandas,
    fix1_AgentSetPandas,
)
from tests.polars.test_agentset_polars import (
    ExampleAgentSetPolars,
    fix2_AgentSetPolars,
)


def get_unique_ids(model: ModelDF) -> pl.Series:
    pandas_set = model.get_agents_of_type(model.agent_types[0])
    polars_set = model.get_agents_of_type(model.agent_types[1])
    return pl.concat(
        [pl.Series(pandas_set["unique_id"].to_list(), dtype=pl.UInt64), polars_set["unique_id"]]
    )


# This serves otherwise ruff complains about the two fixtures not being used
def not_called():
    fix1_AgentSetPandas()
    fix2_AgentSetPolars()


@tg.typechecked
class TestGridPandas:
    @pytest.fixture
    def model(
        self,
        fix1_AgentSetPandas: ExampleAgentSetPandas,
        fix2_AgentSetPolars: ExampleAgentSetPolars,
    ) -> ModelDF:
        model = ModelDF()
        model.agents.add([fix1_AgentSetPandas, fix2_AgentSetPolars])
        return model

    @pytest.fixture
    def grid_moore(self, model: ModelDF) -> GridPandas:
        space = GridPandas(model, dimensions=[3, 3], capacity=2)
        unique_ids = get_unique_ids(model)
        space.place_agents(agents=unique_ids[[0, 1]], pos=[[0, 0], [1, 1]])
        space.set_cells(
            [[0, 0], [1, 1]], properties={"capacity": [1, 3], "property_0": "value_0"}
        )
        return space

    @pytest.fixture
    def grid_moore_torus(self, model: ModelDF) -> GridPandas:
        space = GridPandas(model, dimensions=[3, 3], capacity=2, torus=True)
        unique_ids = get_unique_ids(model)
        space.place_agents(agents=unique_ids[[0, 1]], pos=[[0, 0], [1, 1]])
        space.set_cells(
            [[0, 0], [1, 1]], properties={"capacity": [1, 3], "property_0": "value_0"}
        )
        return space

    @pytest.fixture
    def grid_von_neumann(self, model: ModelDF) -> GridPandas:
        space = GridPandas(model, dimensions=[3, 3], neighborhood_type="von_neumann")
        unique_ids = get_unique_ids(model)
        space.place_agents(agents=unique_ids[[0, 1]], pos=[[0, 0], [1, 1]])
        return space

    @pytest.fixture
    def grid_hexagonal(self, model: ModelDF) -> GridPandas:
        space = GridPandas(model, dimensions=[10, 10], neighborhood_type="hexagonal")
        unique_ids = get_unique_ids(model)
        space.place_agents(agents=unique_ids[[0, 1]], pos=[[5, 4], [5, 5]])
        return space

    def test___init__(self, model: ModelDF):
        # Test with default parameters
        grid1 = GridPandas(model, dimensions=[3, 3])
        assert isinstance(grid1, GridPandas)
        assert isinstance(grid1.agents, pd.DataFrame)
        assert grid1.agents.empty
        assert isinstance(grid1.cells, pd.DataFrame)
        assert grid1.cells.empty
        assert isinstance(grid1.dimensions, list)
        assert len(grid1.dimensions) == 2
        assert isinstance(grid1.neighborhood_type, str)
        assert grid1.neighborhood_type == "moore"
        assert grid1.remaining_capacity == float("inf")
        assert grid1.model == model

        # Test with capacity = 10
        grid2 = GridPandas(model, dimensions=[3, 3], capacity=10)
        assert grid2.remaining_capacity == (10 * 3 * 3)

        # Test with torus = True
        grid3 = GridPandas(model, dimensions=[3, 3], torus=True)
        assert grid3.torus

        # Test with neighborhood_type = "von_neumann"
        grid4 = GridPandas(model, dimensions=[3, 3], neighborhood_type="von_neumann")
        assert grid4.neighborhood_type == "von_neumann"

        # Test with neighborhood_type = "moore"
        grid5 = GridPandas(model, dimensions=[3, 3], neighborhood_type="moore")
        assert grid5.neighborhood_type == "moore"

        # Test with neighborhood_type = "hexagonal"
        grid6 = GridPandas(model, dimensions=[3, 3], neighborhood_type="hexagonal")
        assert grid6.neighborhood_type == "hexagonal"

    def test_get_cells(self, grid_moore: GridPandas):
        # Test with None (all cells)
        result = grid_moore.get_cells()
        assert isinstance(result, pd.DataFrame)
        assert result.reset_index()["dim_0"].tolist() == [0, 1]
        assert result.reset_index()["dim_1"].tolist() == [0, 1]
        assert result["capacity"].tolist() == [1, 3]
        assert result["property_0"].tolist() == ["value_0", "value_0"]

        # Test with GridCoordinate
        result = grid_moore.get_cells([0, 0])
        assert isinstance(result, pd.DataFrame)
        assert result.reset_index()["dim_0"].tolist() == [0]
        assert result.reset_index()["dim_1"].tolist() == [0]
        assert result["capacity"].tolist() == [1]
        assert result["property_0"].tolist() == ["value_0"]

        # Test with GridCoordinates
        result = grid_moore.get_cells([[0, 0], [1, 1]])
        assert isinstance(result, pd.DataFrame)
        assert result.reset_index()["dim_0"].tolist() == [0, 1]
        assert result.reset_index()["dim_1"].tolist() == [0, 1]
        assert result["capacity"].tolist() == [1, 3]
        assert result["property_0"].tolist() == ["value_0", "value_0"]

    def test_get_directions(
        self,
        grid_moore: GridPandas,
        fix1_AgentSetPandas: ExampleAgentSetPandas,
        fix2_AgentSetPolars: ExampleAgentSetPolars,
    ):
        # Test with GridCoordinate
        dir = grid_moore.get_directions(pos0=[1, 1], pos1=[2, 2])
        assert isinstance(dir, pd.DataFrame)
        assert dir["dim_0"].to_list() == [1]
        assert dir["dim_1"].to_list() == [1]

        # Test with GridCoordinates
        dir = grid_moore.get_directions(pos0=[[0, 0], [2, 2]], pos1=[[1, 2], [1, 1]])
        assert isinstance(dir, pd.DataFrame)
        assert dir["dim_0"].to_list() == [1, -1]
        assert dir["dim_1"].to_list() == [2, -1]

        # Test with missing agents (raises ValueError)
        with pytest.raises(ValueError):
            grid_moore.get_directions(
                agents0=fix1_AgentSetPandas, agents1=fix2_AgentSetPolars
            )

        # Test with IdsLike
        grid_moore.place_agents(fix2_AgentSetPolars, [[0, 1], [0, 2], [1, 0], [1, 2]])
        unique_ids = get_unique_ids(grid_moore.model)
        dir = grid_moore.get_directions(
            agents0=unique_ids[[0, 1]], agents1=unique_ids[[4, 5]]
        )
        assert isinstance(dir, pd.DataFrame)
        assert dir["dim_0"].to_list() == [0, -1]
        assert dir["dim_1"].to_list() == [1, 1]

        # Test with two AgentSetDFs
        grid_moore.place_agents(unique_ids[[2, 3]], [[1, 1], [2, 2]])
        dir = grid_moore.get_directions(
            agents0=fix1_AgentSetPandas, agents1=fix2_AgentSetPolars
        )
        assert isinstance(dir, pd.DataFrame)
        assert dir["dim_0"].to_list() == [0, -1, 0, -1]
        assert dir["dim_1"].to_list() == [1, 1, -1, 0]

        # Test with AgentsDF
        dir = grid_moore.get_directions(
            agents0=grid_moore.model.agents, agents1=grid_moore.model.agents
        )
        assert isinstance(dir, pd.DataFrame)
        assert (dir == 0).all().all()

        # Test with normalize
        dir = grid_moore.get_directions(
            agents0=unique_ids[[0, 1]], agents1=unique_ids[[4, 5]], normalize=True
        )
        # Check if the vectors are normalized (length should be 1)
        assert np.allclose(np.sqrt(dir["dim_0"] ** 2 + dir["dim_1"] ** 2), 1.0)
        # Check specific normalized values
        assert np.allclose(dir["dim_0"].to_list(), [0, -1 / np.sqrt(2)])
        assert np.allclose(dir["dim_1"].to_list(), [1, 1 / np.sqrt(2)])

    def test_get_distances(
        self,
        grid_moore: GridPandas,
        fix1_AgentSetPandas: ExampleAgentSetPandas,
        fix2_AgentSetPolars: ExampleAgentSetPolars,
    ):
        # Test with GridCoordinate
        dist = grid_moore.get_distances(pos0=[1, 1], pos1=[2, 2])
        assert isinstance(dist, pd.DataFrame)
        assert np.allclose(dist["distance"].to_list(), [np.sqrt(2)])

        # Test with GridCoordinates
        dist = grid_moore.get_distances(pos0=[[0, 0], [2, 2]], pos1=[[1, 2], [1, 1]])
        assert isinstance(dist, pd.DataFrame)
        assert np.allclose(dist["distance"].to_list(), [np.sqrt(5), np.sqrt(2)])

        # Test with missing agents (raises ValueError)
        with pytest.raises(ValueError):
            grid_moore.get_distances(
                agents0=fix1_AgentSetPandas, agents1=fix2_AgentSetPolars
            )

        # Test with IdsLike
        grid_moore.place_agents(fix2_AgentSetPolars, [[0, 1], [0, 2], [1, 0], [1, 2]])
        unique_ids = get_unique_ids(grid_moore.model)
        dist = grid_moore.get_distances(
            agents0=unique_ids[[0, 1]], agents1=unique_ids[[4, 5]]
        )
        assert isinstance(dist, pd.DataFrame)
        assert np.allclose(dist["distance"].to_list(), [1.0, np.sqrt(2)])

        # Test with two AgentSetDFs
        grid_moore.place_agents(unique_ids[[2, 3]], [[1, 1], [2, 2]])
        dist = grid_moore.get_distances(
            agents0=fix1_AgentSetPandas, agents1=fix2_AgentSetPolars
        )
        assert isinstance(dist, pd.DataFrame)
        assert np.allclose(dist["distance"].to_list(), [1.0, np.sqrt(2), 1.0, 1.0])

        # Test with AgentsDF
        dist = grid_moore.get_distances(
            agents0=grid_moore.model.agents, agents1=grid_moore.model.agents
        )
        assert (dist == 0).all().all()

    def test_get_neighborhood(
        self,
        grid_moore: GridPandas,
        grid_hexagonal: GridPandas,
        grid_von_neumann: GridPandas,
        grid_moore_torus: GridPandas,
    ):
        # Test with radius = int, pos=GridCoordinate
        neighborhood = grid_moore.get_neighborhood(radius=1, pos=[1, 1])
        assert isinstance(neighborhood, pd.DataFrame)
        assert neighborhood.columns.to_list() == [
            "dim_0",
            "dim_1",
            "radius",
            "dim_0_center",
            "dim_1_center",
        ]
        assert neighborhood.shape == (8, 5)
        assert neighborhood["dim_0"].to_list() == [0, 0, 0, 1, 1, 2, 2, 2]
        assert neighborhood["dim_1"].to_list() == [0, 1, 2, 0, 2, 0, 1, 2]
        assert neighborhood["radius"].to_list() == [1] * 8
        assert neighborhood["dim_0_center"].to_list() == [1] * 8
        assert neighborhood["dim_1_center"].to_list() == [1] * 8

        # Test with Sequence[int], pos=Sequence[GridCoordinate]
        neighborhood = grid_moore.get_neighborhood(radius=[1, 2], pos=[[1, 1], [2, 2]])
        assert isinstance(neighborhood, pd.DataFrame)
        assert neighborhood.shape == (8 + 6, 5)
        assert neighborhood["radius"].sort_values().to_list() == [1] * 11 + [2] * 3
        assert neighborhood["dim_0_center"].sort_values().to_list() == [1] * 8 + [2] * 6
        assert neighborhood["dim_1_center"].sort_values().to_list() == [1] * 8 + [2] * 6
        neighborhood = neighborhood.sort_values(["dim_0", "dim_1"])
        assert neighborhood["dim_0"].to_list() == [0] * 5 + [1] * 4 + [2] * 5
        assert neighborhood["dim_1"].to_list() == [
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
        assert isinstance(neighborhood, pd.DataFrame)
        assert neighborhood.shape == (8, 5)
        assert neighborhood["dim_0"].to_list() == [0, 0, 0, 1, 1, 2, 2, 2]
        assert neighborhood["dim_1"].to_list() == [0, 1, 2, 0, 2, 0, 1, 2]
        assert neighborhood["radius"].to_list() == [1] * 8
        assert neighborhood["dim_0_center"].to_list() == [1] * 8
        assert neighborhood["dim_1_center"].to_list() == [1] * 8

        # Test with agent=Sequence[int], pos=Sequence[GridCoordinate]
        neighborhood = grid_moore.get_neighborhood(
            radius=[1, 2], agents=unique_ids[[0, 1]]
        )
        assert isinstance(neighborhood, pd.DataFrame)
        assert neighborhood.shape == (8 + 6, 5)
        assert neighborhood["radius"].sort_values().to_list() == [1] * 11 + [2] * 3
        assert neighborhood["dim_0_center"].sort_values().to_list() == [1] * 8 + [2] * 6
        assert neighborhood["dim_1_center"].sort_values().to_list() == [1] * 8 + [2] * 6
        neighborhood = neighborhood.sort_values(["dim_0", "dim_1"])
        assert neighborhood["dim_0"].to_list() == [0] * 5 + [1] * 4 + [2] * 5
        assert neighborhood["dim_1"].to_list() == [
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
        neighborhood = grid_moore.get_neighborhood(
            radius=1, pos=[1, 1], include_center=True
        )
        assert isinstance(neighborhood, pd.DataFrame)
        assert neighborhood.shape == (9, 5)
        assert neighborhood["dim_0"].to_list() == [1, 0, 0, 0, 1, 1, 2, 2, 2]
        assert neighborhood["dim_1"].to_list() == [1, 0, 1, 2, 0, 2, 0, 1, 2]
        assert neighborhood["radius"].to_list() == [0] + [1] * 8
        assert neighborhood["dim_0_center"].to_list() == [1] * 9
        assert neighborhood["dim_1_center"].to_list() == [1] * 9

        # Test with torus
        neighborhood = grid_moore_torus.get_neighborhood(radius=1, pos=[0, 0])
        assert isinstance(neighborhood, pd.DataFrame)
        assert neighborhood.shape == (8, 5)
        assert neighborhood["dim_0"].to_list() == [2, 2, 2, 0, 0, 1, 1, 1]
        assert neighborhood["dim_1"].to_list() == [2, 0, 1, 2, 1, 2, 0, 1]
        assert neighborhood["radius"].to_list() == [1] * 8
        assert neighborhood["dim_0_center"].to_list() == [0] * 8
        assert neighborhood["dim_1_center"].to_list() == [0] * 8

        # Test with radius and pos of different length
        with pytest.raises(ValueError):
            neighborhood = grid_moore.get_neighborhood(radius=[1, 2], pos=[1, 1])

        # Test with von_neumann neighborhood
        neighborhood = grid_von_neumann.get_neighborhood(radius=1, pos=[1, 1])
        assert isinstance(neighborhood, pd.DataFrame)
        assert neighborhood.shape == (4, 5)
        assert neighborhood["dim_0"].to_list() == [0, 1, 1, 2]
        assert neighborhood["dim_1"].to_list() == [1, 0, 2, 1]
        assert neighborhood["radius"].to_list() == [1] * 4
        assert neighborhood["dim_0_center"].to_list() == [1] * 4
        assert neighborhood["dim_1_center"].to_list() == [1] * 4

        # Test with hexagonal neighborhood (odd cell [2,1] and even cell [2,2])
        neighborhood = grid_hexagonal.get_neighborhood(
            radius=[2, 3], pos=[[5, 4], [5, 5]]
        )
        assert isinstance(neighborhood, pd.DataFrame)
        assert neighborhood.shape == (
            6 * 2 + 12 * 2 + 18,
            5,
        )  # 6 neighbors for radius 1, 12 for radius 2, 18 for radius 3

        # Sort the neighborhood for consistent ordering
        neighborhood = neighborhood.sort_values(
            ["dim_0_center", "dim_1_center", "radius", "dim_0", "dim_1"]
        ).reset_index(drop=True)

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
            list(zip(neighborhood["dim_0"], neighborhood["dim_1"]))
            == expected_neighbors
        )

    def test_get_neighbors(
        self,
        fix2_AgentSetPolars: ExampleAgentSetPolars,
        grid_moore: GridPandas,
        grid_hexagonal: GridPandas,
        grid_von_neumann: GridPandas,
        grid_moore_torus: GridPandas,
    ):
        # Place agents in the grid
        unique_ids = get_unique_ids(grid_moore.model).sort()
        grid_moore.move_agents(
            unique_ids[[0, 1, 2, 3, 4, 5, 6, 7]],
            [[0, 0], [0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1], [2, 2]],
        )

        # Test with radius = int, pos=GridCoordinate
        neighbors = grid_moore.get_neighbors(radius=1, pos=[1, 1])
        assert isinstance(neighbors, pd.DataFrame)
        assert neighbors.index.name == "agent_id"
        assert neighbors.columns.to_list() == ["dim_0", "dim_1"]
        assert neighbors.shape == (8, 2)
        assert neighbors["dim_0"].to_list() == [0, 0, 0, 1, 1, 2, 2, 2]
        assert neighbors["dim_1"].to_list() == [0, 1, 2, 0, 2, 0, 1, 2]
        assert set(neighbors.index) == set(unique_ids[[0, 1, 2, 3, 4, 5, 6, 7]])

        # Test with Sequence[int], pos=Sequence[GridCoordinate]
        neighbors = grid_moore.get_neighbors(radius=[1, 2], pos=[[1, 1], [2, 2]])
        assert isinstance(neighbors, pd.DataFrame)
        assert neighbors.index.name == "agent_id"
        assert neighbors.shape == (8, 2)
        neighbors = neighbors.sort_values(["dim_0", "dim_1"])
        assert neighbors["dim_0"].to_list() == [0, 0, 0, 1, 1, 2, 2, 2]
        assert neighbors["dim_1"].to_list() == [0, 1, 2, 0, 2, 0, 1, 2]
        assert set(neighbors.index) == set(unique_ids[[0, 1, 2, 3, 4, 5, 6, 7]])

        # Test with agent=int
        neighbors = grid_moore.get_neighbors(radius=1, agents=unique_ids[0])
        assert isinstance(neighbors, pd.DataFrame)
        assert neighbors.index.name == "agent_id"
        assert neighbors.shape == (2, 2)
        assert neighbors["dim_0"].to_list() == [0, 1]
        assert neighbors["dim_1"].to_list() == [1, 0]
        assert set(neighbors.index) == set(unique_ids[[1, 3]])

        # Test with agent=Sequence[int]
        neighbors = grid_moore.get_neighbors(radius=[1, 2], agents=unique_ids[[0, 7]])
        assert isinstance(neighbors, pd.DataFrame)
        assert neighbors.index.name == "agent_id"
        assert neighbors.shape == (7, 2)
        neighbors = neighbors.sort_values(["dim_0", "dim_1"])
        assert neighbors["dim_0"].to_list() == [0, 0, 0, 1, 1, 2, 2]
        assert neighbors["dim_1"].to_list() == [0, 1, 2, 0, 2, 0, 1]
        assert set(neighbors.index) == set(unique_ids[[0, 1, 2, 3, 4, 5, 6]])

        # Test with include_center
        neighbors = grid_moore.get_neighbors(radius=1, pos=[1, 1], include_center=True)
        assert isinstance(neighbors, pd.DataFrame)
        assert neighbors.index.name == "agent_id"
        assert neighbors.shape == (8, 2)  # No agent at [1, 1], so still 8 neighbors
        assert neighbors["dim_0"].to_list() == [0, 0, 0, 1, 1, 2, 2, 2]
        assert neighbors["dim_1"].to_list() == [0, 1, 2, 0, 2, 0, 1, 2]
        assert set(neighbors.index) == set(unique_ids[[0, 1, 2, 3, 4, 5, 6, 7]])

        # Test with torus
        grid_moore_torus.move_agents(
            unique_ids[[0, 1, 2, 3, 4, 5, 6, 7]],
            [[2, 2], [2, 0], [2, 1], [0, 2], [0, 1], [1, 2], [1, 0], [1, 1]],
        )
        neighbors = grid_moore_torus.get_neighbors(radius=1, pos=[0, 0])
        assert isinstance(neighbors, pd.DataFrame)
        assert neighbors.index.name == "agent_id"
        assert neighbors.shape == (8, 2)
        assert neighbors["dim_0"].to_list() == [2, 2, 2, 0, 0, 1, 1, 1]
        assert neighbors["dim_1"].to_list() == [2, 0, 1, 2, 1, 2, 0, 1]
        assert set(neighbors.index) == set(unique_ids[0, 1, 2, 3, 4, 5, 6, 7])

        # Test with radius and pos of different length
        with pytest.raises(ValueError):
            neighbors = grid_moore.get_neighbors(radius=[1, 2], pos=[1, 1])

        # Test with von_neumann neighborhood
        grid_von_neumann.move_agents(
            unique_ids[[0, 1, 2, 3]], [[0, 1], [1, 0], [1, 2], [2, 1]]
        )
        neighbors = grid_von_neumann.get_neighbors(radius=1, pos=[1, 1])
        assert isinstance(neighbors, pd.DataFrame)
        assert neighbors.index.name == "agent_id"
        assert neighbors.shape == (4, 2)
        assert neighbors["dim_0"].to_list() == [0, 1, 1, 2]
        assert neighbors["dim_1"].to_list() == [1, 0, 2, 1]
        assert set(neighbors.index) == set(unique_ids[[0, 1, 2, 3]])

        # Test with hexagonal neighborhood (odd cell [5,4] and even cell [5,5])
        grid_hexagonal.move_agents(
            unique_ids[range(8)],
            [[4, 4], [4, 5], [5, 3], [5, 5], [6, 3], [6, 4], [5, 4], [5, 6]],
        )
        neighbors = grid_hexagonal.get_neighbors(radius=[2, 3], pos=[[5, 4], [5, 5]])
        assert isinstance(neighbors, pd.DataFrame)
        assert neighbors.index.name == "agent_id"
        assert neighbors.shape == (8, 2)  # All agents are within the neighborhood

        # Sort the neighbors for consistent ordering
        neighbors = neighbors.sort_values(["dim_0", "dim_1"])

        assert neighbors["dim_0"].to_list() == [
            4,
            4,
            5,
            5,
            5,
            5,
            6,
            6,
        ]
        assert neighbors["dim_1"].to_list() == [4, 5, 3, 4, 5, 6, 3, 4]
        assert set(neighbors.index) == set(unique_ids[range(8)])

    def test_is_available(self, grid_moore: GridPandas):
        # Test with GridCoordinate
        result = grid_moore.is_available([0, 0])
        assert isinstance(result, pd.DataFrame)
        assert result["available"].tolist() == [False]
        result = grid_moore.is_available([1, 1])
        assert result["available"].tolist() == [True]

        # Test with GridCoordinates
        result = grid_moore.is_available([[0, 0], [1, 1]])
        assert result["available"].tolist() == [False, True]

    def test_is_empty(self, grid_moore: GridPandas):
        # Test with GridCoordinate
        result = grid_moore.is_empty([0, 0])
        assert isinstance(result, pd.DataFrame)
        assert result["empty"].tolist() == [False]
        result = grid_moore.is_empty([1, 1])
        assert result["empty"].tolist() == [False]

        # Test with GridCoordinates
        result = grid_moore.is_empty([[0, 0], [1, 1]])
        assert result["empty"].tolist() == [False, False]

    def test_is_full(self, grid_moore: GridPandas):
        # Test with GridCoordinate
        result = grid_moore.is_full([0, 0])
        assert isinstance(result, pd.DataFrame)
        assert result["full"].tolist() == [True]
        result = grid_moore.is_full([1, 1])
        assert result["full"].tolist() == [False]

        # Test with GridCoordinates
        result = grid_moore.is_full([[0, 0], [1, 1]])
        assert result["full"].tolist() == [True, False]

    def test_move_agents(
        self,
        grid_moore: GridPandas,
        fix1_AgentSetPandas: ExampleAgentSetPandas,
        fix2_AgentSetPolars: ExampleAgentSetPolars,
    ):
        # Test with IdsLike
        unique_ids = get_unique_ids(grid_moore.model)
        space = grid_moore.move_agents(agents=unique_ids[1], pos=[1, 1], inplace=False)
        assert space.remaining_capacity == (2 * 3 * 3 - 2)
        assert len(space.agents) == 2
        # reorder the agents according to the original order
        agents = space.agents.reindex(unique_ids).dropna()
        assert agents.index.to_list() == unique_ids[[0, 1]].to_list()
        assert agents["dim_0"].to_list() == [0, 1]
        assert agents["dim_1"].to_list() == [0, 1]

        # Test with AgentSetDF
        with pytest.warns(RuntimeWarning):
            space = grid_moore.move_agents(
                agents=fix2_AgentSetPolars,
                pos=[[0, 0], [1, 0], [2, 0], [0, 1]],
                inplace=False,
            )
            unique_ids = get_unique_ids(space.model)
            assert space.remaining_capacity == (2 * 3 * 3 - 6)
            assert len(space.agents) == 6
            agents = space.agents.reindex(unique_ids).dropna()
            assert agents.index.to_list() == unique_ids[[0, 1, 4, 5, 6, 7]].to_list()
            assert agents["dim_0"].to_list() == [0, 1, 0, 1, 2, 0]
            assert agents["dim_1"].to_list() == [0, 1, 0, 0, 0, 1]

        # Test with Collection[AgentSetDF]
        with pytest.warns(RuntimeWarning):
            space = grid_moore.move_agents(
                agents=[fix1_AgentSetPandas, fix2_AgentSetPolars],
                pos=[[0, 2], [1, 2], [2, 2], [0, 1], [1, 1], [2, 1], [0, 0], [1, 0]],
                inplace=False,
            )
        unique_ids = get_unique_ids(grid_moore.model)
        assert space.remaining_capacity == (2 * 3 * 3 - 8)
        assert len(space.agents) == 8
        agents = space.agents.reindex(unique_ids).dropna()
        assert agents.index.to_list() == unique_ids[[0, 1, 2, 3, 4, 5, 6, 7]].to_list()
        assert agents["dim_0"].to_list() == [0, 1, 2, 0, 1, 2, 0, 1]
        assert agents["dim_1"].to_list() == [2, 2, 2, 1, 1, 1, 0, 0]

        # Raises ValueError if len(agents) != len(pos)
        with pytest.raises(ValueError):
            space = grid_moore.move_agents(
                agents=unique_ids[[0, 1]], pos=[[0, 0], [1, 1], [2, 2]], inplace=False
            )

        # Test with AgentsDF, pos=DataFrame
        pos = pd.DataFrame(
            {
                "unaligned_index": range(1000, 1008),
                "dim_0": [0, 1, 2, 0, 1, 2, 0, 1],
                "dim_1": [2, 2, 2, 1, 1, 1, 0, 0],
            }
        ).set_index("unaligned_index")

        with pytest.warns(RuntimeWarning):
            space = grid_moore.move_agents(
                agents=grid_moore.model.agents,
                pos=pos,
                inplace=False,
            )
        assert space.remaining_capacity == (2 * 3 * 3 - 8)
        assert len(space.agents) == 8
        agents = space.agents.reindex(unique_ids).dropna()
        assert agents.index.to_list() == unique_ids[[0, 1, 2, 3, 4, 5, 6, 7]].to_list()
        assert agents["dim_0"].to_list() == [0, 1, 2, 0, 1, 2, 0, 1]
        assert agents["dim_1"].to_list() == [2, 2, 2, 1, 1, 1, 0, 0]

        # Test with agents=int, pos=DataFrame
        pos = pd.DataFrame({"dim_0": [0], "dim_1": [2]})
        space = grid_moore.move_agents(agents=unique_ids[1], pos=pos, inplace=False)
        assert space.remaining_capacity == (2 * 3 * 3 - 2)
        assert len(space.agents) == 2
        agents = space.agents.reindex(unique_ids).dropna()
        assert agents.index.to_list() == unique_ids[[0, 1]].to_list()
        assert agents["dim_0"].to_list() == [0, 0]
        assert agents["dim_1"].to_list() == [0, 2]

    def test_move_to_available(self, grid_moore: GridPandas):
        # Test with GridCoordinate
        last = None
        different = False
        for _ in range(10):
            available_cells = grid_moore.available_cells
            space = grid_moore.move_to_available(0, inplace=False)
            if last is not None and not different:
                if (space.agents[["dim_0", "dim_1"]].values != last).any():
                    different = True
            assert space.agents[["dim_0", "dim_1"]].values[0] in available_cells.values
            last = space.agents[["dim_0", "dim_1"]].values
        assert different

        # Test with GridCoordinates
        last = None
        different = False
        for _ in range(10):
            available_cells = grid_moore.available_cells
            space = grid_moore.move_to_available([0, 1], inplace=False)
            if last is not None and not different:
                if (space.agents[["dim_0", "dim_1"]].values != last).any():
                    different = True
            assert (
                space.agents[["dim_0", "dim_1"]].values[0] in available_cells.values
            ) and (space.agents[["dim_0", "dim_1"]].values[1] in available_cells.values)
            last = space.agents[["dim_0", "dim_1"]].values
        assert different

        # Test with AgentSetDF
        last = None
        different = False
        for _ in range(10):
            available_cells = grid_moore.available_cells
            space = grid_moore.move_to_available(grid_moore.model.agents, inplace=False)
            if last is not None and not different:
                if (space.agents["dim_0"].values != last).any():
                    different = True
            assert (
                space.agents[["dim_0", "dim_1"]].values[0] in available_cells.values
            ) and (space.agents[["dim_0", "dim_1"]].values[1] in available_cells.values)
            last = space.agents["dim_0"].values
        assert different

    def test_move_to_empty(self, grid_moore: GridPandas):
        # Test with GridCoordinate
        last = None
        different = False
        for _ in range(10):
            empty_cells = grid_moore.empty_cells
            space = grid_moore.move_to_empty(0, inplace=False)
            if last is not None and not different:
                if (space.agents[["dim_0", "dim_1"]].values != last).any():
                    different = True
            assert space.agents[["dim_0", "dim_1"]].values[0] in empty_cells.values
            last = space.agents[["dim_0", "dim_1"]].values
        assert different

        # Test with GridCoordinates
        last = None
        different = False
        for _ in range(10):
            empty_cells = grid_moore.empty_cells
            space = grid_moore.move_to_empty([0, 1], inplace=False)
            if last is not None and not different:
                if (space.agents[["dim_0", "dim_1"]].values != last).any():
                    different = True
            assert (
                space.agents[["dim_0", "dim_1"]].values[0] in empty_cells.values
            ) and (space.agents[["dim_0", "dim_1"]].values[1] in empty_cells.values)
            last = space.agents[["dim_0", "dim_1"]].values
        assert different

        # Test with AgentSetDF
        last = None
        different = False
        for _ in range(10):
            empty_cells = grid_moore.empty_cells
            space = grid_moore.move_to_empty(grid_moore.model.agents, inplace=False)
            if last is not None and not different:
                if (space.agents["dim_0"].values != last).any():
                    different = True
            assert (
                space.agents[["dim_0", "dim_1"]].values[0] in empty_cells.values
            ) and (space.agents[["dim_0", "dim_1"]].values[1] in empty_cells.values)
            last = space.agents["dim_0"].values
        assert different

    def test_out_of_bounds(self, grid_moore: GridPandas):
        # Test with GridCoordinate
        out_of_bounds = grid_moore.out_of_bounds([11, 11])
        assert isinstance(out_of_bounds, pd.DataFrame)
        assert out_of_bounds.shape == (1, 3)
        assert out_of_bounds.columns.to_list() == ["dim_0", "dim_1", "out_of_bounds"]
        assert out_of_bounds.iloc[0].to_list() == [11, 11, True]

        # Test with GridCoordinates
        out_of_bounds = grid_moore.out_of_bounds([[0, 0], [11, 11]])
        assert isinstance(out_of_bounds, pd.DataFrame)
        assert out_of_bounds.shape == (2, 3)
        assert out_of_bounds.columns.to_list() == ["dim_0", "dim_1", "out_of_bounds"]
        assert out_of_bounds.iloc[0].to_list() == [0, 0, False]
        assert out_of_bounds.iloc[1].to_list() == [11, 11, True]

    def test_place_agents(
        self,
        grid_moore: GridPandas,
        fix1_AgentSetPandas: ExampleAgentSetPandas,
        fix2_AgentSetPolars: ExampleAgentSetPolars,
    ):
        unique_ids = get_unique_ids(grid_moore.model)
        # Test with IdsLike
        with pytest.warns(RuntimeWarning):
            space = grid_moore.place_agents(
                agents=unique_ids[[1, 2]], pos=[[1, 1], [2, 2]], inplace=False
            )
        assert space.remaining_capacity == (2 * 3 * 3 - 3)
        assert len(space.agents) == 3
        assert space.agents.index.to_list() == [0, 1, 2]
        assert space.agents["dim_0"].to_list() == [0, 1, 2]
        assert space.agents["dim_1"].to_list() == [0, 1, 2]

        # Test with agents not in the model
        with pytest.raises(ValueError):
            space = grid_moore.place_agents(
                agents=unique_ids[[10, 11]],
                pos=[[0, 0], [1, 0]],
                inplace=False,
            )

        # Test with AgentSetDF
        space = grid_moore.place_agents(
            agents=fix2_AgentSetPolars,
            pos=[[0, 0], [1, 0], [2, 0], [0, 1]],
            inplace=False,
        )
        assert space.remaining_capacity == (2 * 3 * 3 - 6)
        assert len(space.agents) == 6
        assert space.agents.index.to_list() == unique_ids[[0, 1, 4, 5, 6, 7]]
        assert space.agents["dim_0"].to_list() == [0, 1, 0, 1, 2, 0]
        assert space.agents["dim_1"].to_list() == [0, 1, 0, 0, 0, 1]

        # Test with Collection[AgentSetDF]
        with pytest.warns(RuntimeWarning):
            space = grid_moore.place_agents(
                agents=[fix1_AgentSetPandas, fix2_AgentSetPolars],
                pos=[[0, 2], [1, 2], [2, 2], [0, 1], [1, 1], [2, 1], [0, 0], [1, 0]],
                inplace=False,
            )
        assert space.remaining_capacity == (2 * 3 * 3 - 8)
        assert len(space.agents) == 8
        assert space.agents.index.to_list() == unique_ids[[0, 1, 2, 3, 4, 5, 6, 7]]
        assert space.agents["dim_0"].to_list() == [0, 1, 2, 0, 1, 2, 0, 1]
        assert space.agents["dim_1"].to_list() == [2, 2, 2, 1, 1, 1, 0, 0]

        # Test with AgentsDF, pos=DataFrame
        pos = pd.DataFrame(
            {
                "unaligned_index": range(1000, 1008),
                "dim_0": [0, 1, 2, 0, 1, 2, 0, 1],
                "dim_1": [2, 2, 2, 1, 1, 1, 0, 0],
            }
        ).set_index("unaligned_index")

        with pytest.warns(RuntimeWarning):
            space = grid_moore.place_agents(
                agents=grid_moore.model.agents,
                pos=pos,
                inplace=False,
            )
        assert space.remaining_capacity == (2 * 3 * 3 - 8)
        assert len(space.agents) == 8
        assert space.agents.index.to_list() == [0, 1, 2, 3, 4, 5, 6, 7]
        assert space.agents["dim_0"].to_list() == [0, 1, 2, 0, 1, 2, 0, 1]
        assert space.agents["dim_1"].to_list() == [2, 2, 2, 1, 1, 1, 0, 0]

        # Test with agents=int, pos=DataFrame
        pos = pd.DataFrame({"dim_0": [0], "dim_1": [2]})
        with pytest.warns(RuntimeWarning):
            space = grid_moore.place_agents(
                agents=unique_ids[1], pos=pos, inplace=False
            )
        assert space.remaining_capacity == (2 * 3 * 3 - 2)
        assert len(space.agents) == 2
        assert space.agents.index.to_list() == unique_ids[[0, 1]]
        assert space.agents["dim_0"].to_list() == [0, 0]
        assert space.agents["dim_1"].to_list() == [0, 2]

    def test_place_to_available(self, grid_moore: GridPandas):
        # Test with GridCoordinate
        last = None
        different = False
        for _ in range(10):
            available_cells = grid_moore.available_cells
            space = grid_moore.place_to_available(0, inplace=False)
            if last is not None and not different:
                if (space.agents[["dim_0", "dim_1"]].values != last).any():
                    different = True
            assert space.agents[["dim_0", "dim_1"]].values[0] in available_cells.values
            last = space.agents[["dim_0", "dim_1"]].values
        assert different
        # Test with GridCoordinates
        last = None
        different = False
        for _ in range(10):
            available_cells = grid_moore.available_cells
            space = grid_moore.place_to_available([0, 1], inplace=False)
            if last is not None and not different:
                if (space.agents[["dim_0", "dim_1"]].values != last).any():
                    different = True
            assert (
                space.agents[["dim_0", "dim_1"]].values[0] in available_cells.values
            ) and (space.agents[["dim_0", "dim_1"]].values[1] in available_cells.values)
            last = space.agents[["dim_0", "dim_1"]].values
        assert different
        # Test with AgentSetDF
        last = None
        different = False
        for _ in range(10):
            available_cells = grid_moore.available_cells
            space = grid_moore.place_to_available(
                grid_moore.model.agents, inplace=False
            )
            if last is not None and not different:
                if (space.agents["dim_0"].values != last).any():
                    different = True
            assert (
                space.agents[["dim_0", "dim_1"]].values[0] in available_cells.values
            ) and (space.agents[["dim_0", "dim_1"]].values[1] in available_cells.values)
            last = space.agents["dim_0"].values
        assert different

    def test_place_to_empty(self, grid_moore: GridPandas):
        # Test with GridCoordinate
        last = None
        different = False
        for _ in range(10):
            empty_cells = grid_moore.empty_cells
            space = grid_moore.place_to_empty(0, inplace=False)
            if last is not None and not different:
                if (space.agents[["dim_0", "dim_1"]].values != last).any():
                    different = True
            assert space.agents[["dim_0", "dim_1"]].values[0] in empty_cells.values
            last = space.agents[["dim_0", "dim_1"]].values
        assert different
        # Test with GridCoordinates
        last = None
        different = False
        for _ in range(10):
            empty_cells = grid_moore.empty_cells
            space = grid_moore.place_to_empty([0, 1], inplace=False)
            if last is not None and not different:
                if (space.agents[["dim_0", "dim_1"]].values != last).any():
                    different = True
            assert (
                space.agents[["dim_0", "dim_1"]].values[0] in empty_cells.values
            ) and (space.agents[["dim_0", "dim_1"]].values[1] in empty_cells.values)
            last = space.agents[["dim_0", "dim_1"]].values
        assert different
        # Test with AgentSetDF
        last = None
        different = False
        for _ in range(10):
            empty_cells = grid_moore.empty_cells
            space = grid_moore.place_to_empty(grid_moore.model.agents, inplace=False)
            if last is not None and not different:
                if (space.agents["dim_0"].values != last).any():
                    different = True
            assert (
                space.agents[["dim_0", "dim_1"]].values[0] in empty_cells.values
            ) and (space.agents[["dim_0", "dim_1"]].values[1] in empty_cells.values)
            last = space.agents["dim_0"].values
        assert different

    def test_random_agents(self, grid_moore: GridPandas):
        different = False
        agents0 = grid_moore.random_agents(1)
        for _ in range(100):
            agents1 = grid_moore.random_agents(1)
            if (agents0.values != agents1.values).all().all():
                different = True
                break
        assert different

    def test_random_pos(self, grid_moore: GridPandas):
        different = False
        last = None
        for _ in range(10):
            random_pos = grid_moore.random_pos(5)
            assert isinstance(random_pos, pd.DataFrame)
            assert len(random_pos) == 5
            assert random_pos.columns.to_list() == ["dim_0", "dim_1"]
            assert not grid_moore.out_of_bounds(random_pos)["out_of_bounds"].any()
            if last is not None and not different:
                if (last != random_pos).any().any():
                    different = True
                break
            last = random_pos
        assert different

    def test_remove_agents(
        self,
        grid_moore: GridPandas,
        fix1_AgentSetPandas: ExampleAgentSetPandas,
        fix2_AgentSetPolars: ExampleAgentSetPolars,
    ):
        grid_moore.move_agents(
            [0, 1, 2, 3, 4, 5, 6, 7],
            [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]],
        )
        capacity = grid_moore.remaining_capacity
        # Test with IdsLike
        space = grid_moore.remove_agents([1, 2], inplace=False)
        assert space.agents.shape == (6, 2)
        assert space.remaining_capacity == capacity + 2
        assert space.agents.index.to_list() == [0, 3, 4, 5, 6, 7]
        assert [
            x for id in space.model.agents.index.values() for x in id.to_list()
        ] == [x for x in range(8)]

        # Test with AgentSetDF
        space = grid_moore.remove_agents(fix1_AgentSetPandas, inplace=False)
        assert space.agents.shape == (4, 2)
        assert space.remaining_capacity == capacity + 4
        assert space.agents.index.to_list() == [4, 5, 6, 7]
        assert [
            x for id in space.model.agents.index.values() for x in id.to_list()
        ] == [x for x in range(8)]

        # Test with Collection[AgentSetDF]
        space = grid_moore.remove_agents(
            [fix1_AgentSetPandas, fix2_AgentSetPolars], inplace=False
        )
        assert [
            x for id in space.model.agents.index.values() for x in id.to_list()
        ] == [x for x in range(8)]
        assert space.agents.empty
        assert space.remaining_capacity == capacity + 8
        # Test with AgentsDF
        space = grid_moore.remove_agents(grid_moore.model.agents, inplace=False)
        assert space.remaining_capacity == capacity + 8
        assert space.agents.empty
        assert [
            x for id in space.model.agents.index.values() for x in id.to_list()
        ] == [x for x in range(8)]

    def test_sample_cells(self, grid_moore: GridPandas, model: ModelDF):
        # Test with default parameters
        replacement = False
        same = True
        last = None
        for _ in range(10):
            result = grid_moore.sample_cells(10)
            assert len(result) == 10
            assert isinstance(result, pd.DataFrame)
            assert result.columns.to_list() == ["dim_0", "dim_1"]
            counts = result.groupby(result.columns.to_list()).size()
            assert (counts <= 2).all()
            if not replacement and (counts > 1).any():
                replacement = True
            if same and last is not None:
                same = (result == last).all().all()
            if not same and replacement:
                break
            last = result
        assert replacement and not same

        # Test with too many samples
        with pytest.raises(AssertionError):
            grid_moore.sample_cells(100)

        # Test with 'empty' cell_type

        result = grid_moore.sample_cells(14, cell_type="empty")
        assert len(result) == 14
        assert isinstance(result, pd.DataFrame)
        assert result.columns.to_list() == ["dim_0", "dim_1"]
        counts = result.groupby(result.columns.to_list()).size()

        ## (0, 1) and (1, 1) are not in the result
        assert not ((result["dim_0"] == 0) & (result["dim_1"] == 0)).any(), (
            "Found (0, 1) in the result"
        )
        assert not ((result["dim_0"] == 1) & (result["dim_1"] == 1)).any(), (
            "Found (1, 1) in the result"
        )

        # 14 should be the max number of empty cells
        with pytest.raises(AssertionError):
            grid_moore.sample_cells(15, cell_type="empty")

        # Test with 'available' cell_type
        result = grid_moore.sample_cells(16, cell_type="available")
        assert len(result) == 16
        assert isinstance(result, pd.DataFrame)
        assert result.columns.to_list() == ["dim_0", "dim_1"]
        counts = result.groupby(result.columns.to_list()).size()

        # 16 should be the max number of available cells
        with pytest.raises(AssertionError):
            grid_moore.sample_cells(17, cell_type="available")

        # Test with 'full' cell_type and no replacement
        grid_moore.set_cells([[0, 0], [1, 1]], properties={"capacity": 1})
        result = grid_moore.sample_cells(2, cell_type="full", with_replacement=False)
        assert len(result) == 2
        assert isinstance(result, pd.DataFrame)
        assert result.columns.to_list() == ["dim_0", "dim_1"]
        assert (
            ((result["dim_0"] == 0) & (result["dim_1"] == 0))
            | ((result["dim_0"] == 1) & (result["dim_1"] == 1))
        ).all()
        # 2 should be the max number of full cells
        with pytest.raises(AssertionError):
            grid_moore.sample_cells(3, cell_type="full", with_replacement=False)

        # Test with grid with infinite capacity
        grid_moore = GridPandas(model, dimensions=[3, 3], capacity=np.inf)
        result = grid_moore.sample_cells(10)
        assert len(result) == 10
        assert isinstance(result, pd.DataFrame)
        assert result.columns.to_list() == ["dim_0", "dim_1"]

    def test_set_cells(self, model: ModelDF):
        grid_moore = GridPandas(model, dimensions=[3, 3], capacity=2)

        # Test with GridCoordinate
        grid_moore.set_cells(
            [0, 0], properties={"capacity": 1, "property_0": "value_0"}
        )
        assert grid_moore.remaining_capacity == (2 * 3 * 3 - 1)
        cell_df = grid_moore.get_cells([0, 0])
        assert cell_df.iloc[0]["capacity"] == 1
        assert cell_df.iloc[0]["property_0"] == "value_0"

        # Test with GridCoordinates
        grid_moore.set_cells(
            [[1, 1], [2, 2]], properties={"capacity": 3, "property_1": "value_1"}
        )
        assert grid_moore.remaining_capacity == (2 * 3 * 3 - 1 + 2)
        cell_df = grid_moore.get_cells([[1, 1], [2, 2]])
        assert cell_df.iloc[0]["capacity"] == 3
        assert cell_df.iloc[0]["property_1"] == "value_1"
        assert cell_df.iloc[1]["capacity"] == 3
        assert cell_df.iloc[1]["property_1"] == "value_1"
        cell_df = grid_moore.get_cells([0, 0])
        assert cell_df.iloc[0]["capacity"] == 1
        assert cell_df.iloc[0]["property_0"] == "value_0"

        # Test with DataFrame with dimensions as columns
        df = pd.DataFrame(
            {"dim_0": [0, 1, 2], "dim_1": [0, 1, 2], "capacity": [2, 2, 2]}
        )
        grid_moore.set_cells(df)
        assert grid_moore.remaining_capacity == (2 * 3 * 3)

        cells_df = grid_moore.get_cells([[0, 0], [1, 1], [2, 2]])

        assert cells_df.iloc[0]["capacity"] == 2
        assert cells_df.iloc[1]["capacity"] == 2
        assert cells_df.iloc[2]["capacity"] == 2
        assert cells_df.iloc[0]["property_0"] == "value_0"
        assert cells_df.iloc[1]["property_1"] == "value_1"
        assert cells_df.iloc[2]["property_1"] == "value_1"

        # Test with DataFrame without capacity
        df = pd.DataFrame(
            {"dim_0": [0, 1, 2], "dim_1": [0, 1, 2], "property_2": [0, 1, 2]}
        )
        grid_moore.set_cells(df)
        assert grid_moore.remaining_capacity == (2 * 3 * 3)
        assert grid_moore.get_cells([[0, 0], [1, 1], [2, 2]])[
            "property_2"
        ].to_list() == [0, 1, 2]

        # Test with DataFrame with dimensions as index
        df = pd.DataFrame(
            {"capacity": [1, 1, 1]},
            index=pd.MultiIndex.from_tuples(
                [(0, 0), (1, 1), (2, 2)], names=["dim_0", "dim_1"]
            ),
        )
        space = grid_moore.set_cells(df, inplace=False)
        assert space.remaining_capacity == (2 * 3 * 3 - 3)

        cells_df = space.get_cells([[0, 0], [1, 1], [2, 2]])
        assert cells_df.iloc[0]["capacity"] == 1
        assert cells_df.iloc[1]["capacity"] == 1
        assert cells_df.iloc[2]["capacity"] == 1
        assert cells_df.iloc[0]["property_0"] == "value_0"
        assert cells_df.iloc[1]["property_1"] == "value_1"
        assert cells_df.iloc[2]["property_1"] == "value_1"

        # Add 2 agents to a cell, then set the cell capacity to 1
        unique_ids = get_unique_ids(grid_moore.model)
        grid_moore.place_agents(unique_ids[1, 2], [[0, 0], [0, 0]])
        with pytest.raises(AssertionError):
            grid_moore.set_cells([0, 0], properties={"capacity": 1})

    def test_swap_agents(
        self,
        grid_moore: GridPandas,
        fix1_AgentSetPandas: ExampleAgentSetPandas,
        fix2_AgentSetPolars: ExampleAgentSetPolars,
    ):
        grid_moore.move_agents(
            [0, 1, 2, 3, 4, 5, 6, 7],
            [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]],
        )
        # Test with IdsLike
        space = grid_moore.swap_agents([0, 1], [2, 3], inplace=False)
        assert space.agents.loc[0].tolist() == grid_moore.agents.loc[2].tolist()
        assert space.agents.loc[1].tolist() == grid_moore.agents.loc[3].tolist()
        assert space.agents.loc[2].tolist() == grid_moore.agents.loc[0].tolist()
        assert space.agents.loc[3].tolist() == grid_moore.agents.loc[1].tolist()
        # Test with AgentSetDFs
        space = grid_moore.swap_agents(
            fix1_AgentSetPandas, fix2_AgentSetPolars, inplace=False
        )
        assert space.agents.loc[0].to_list() == grid_moore.agents.loc[4].to_list()
        assert space.agents.loc[1].to_list() == grid_moore.agents.loc[5].to_list()
        assert space.agents.loc[2].to_list() == grid_moore.agents.loc[6].to_list()
        assert space.agents.loc[3].tolist() == grid_moore.agents.loc[7].tolist()

    def test_torus_adj(self, grid_moore: GridPandas, grid_moore_torus: GridPandas):
        # Test with non-toroidal grid
        with pytest.raises(ValueError):
            grid_moore.torus_adj([10, 10])

        # Test with toroidal grid (GridCoordinate)
        adj_df = grid_moore_torus.torus_adj([10, 8])
        assert isinstance(adj_df, pd.DataFrame)
        assert adj_df.shape == (1, 2)
        assert adj_df.columns.to_list() == ["dim_0", "dim_1"]
        assert adj_df.iloc[0].to_list() == [1, 2]

        # Test with toroidal grid (GridCoordinates)
        adj_df = grid_moore_torus.torus_adj([[10, 8], [15, 11]])
        assert isinstance(adj_df, pd.DataFrame)
        assert adj_df.shape == (2, 2)
        assert adj_df.columns.to_list() == ["dim_0", "dim_1"]
        assert adj_df.iloc[0].to_list() == [1, 2]
        assert adj_df.iloc[1].to_list() == [0, 2]

    def test___getitem__(self, grid_moore: GridPandas):
        # Test out of bounds
        with pytest.raises(ValueError):
            grid_moore[[5, 5]]

        # Test with GridCoordinate
        df = grid_moore[[0, 0]]
        assert isinstance(df, pd.DataFrame)
        assert df.index.names == ["dim_0", "dim_1"]
        assert df.index.to_list() == [(0, 0)]
        assert df.columns.to_list() == ["capacity", "property_0", "agent_id"]
        assert df.iloc[0].to_list() == [1, "value_0", 0]

        # Test with GridCoordinates
        df = grid_moore[[[0, 0], [1, 1]]]
        assert isinstance(df, pd.DataFrame)
        assert df.index.names == ["dim_0", "dim_1"]
        assert df.index.to_list() == [(0, 0), (1, 1)]
        assert df.columns.to_list() == ["capacity", "property_0", "agent_id"]
        assert df.iloc[0].to_list() == [1, "value_0", 0]
        assert df.iloc[1].to_list() == [3, "value_0", 1]

    def test___setitem__(self, grid_moore: GridPandas):
        # Test with out-of-bounds
        with pytest.raises(ValueError):
            grid_moore[[5, 5]] = {"capacity": 10}

        # Test with GridCoordinate
        grid_moore[[0, 0]] = {"capacity": 10}
        assert grid_moore.get_cells([[0, 0]]).iloc[0]["capacity"] == 10
        # Test with GridCoordinates
        grid_moore[[[0, 0], [1, 1]]] = {"capacity": 20}
        assert grid_moore.get_cells([[0, 0], [1, 1]])["capacity"].tolist() == [20, 20]

    # Property tests
    def test_agents(self, grid_moore: GridPandas):
        unique_ids = get_unique_ids(grid_moore.model)
        assert isinstance(grid_moore.agents, pd.DataFrame)
        assert grid_moore.agents.index.name == "agent_id"
        assert grid_moore.agents.index.to_list() == unique_ids[[0, 1]]
        assert grid_moore.agents.columns.to_list() == ["dim_0", "dim_1"]
        assert grid_moore.agents["dim_0"].to_list() == [0, 1]
        assert grid_moore.agents["dim_1"].to_list() == [0, 1]

    def test_available_cells(self, grid_moore: GridPandas):
        result = grid_moore.available_cells
        assert len(result) == 8
        assert isinstance(result, pd.DataFrame)
        assert result.columns.to_list() == ["dim_0", "dim_1"]

    def test_cells(self, grid_moore: GridPandas):
        result = grid_moore.cells
        assert isinstance(result, pd.DataFrame)
        assert result.index.names == ["dim_0", "dim_1"]
        assert result.columns.to_list() == ["capacity", "property_0", "agent_id"]
        assert result.index.to_list() == [(0, 0), (1, 1)]
        assert result["capacity"].to_list() == [1, 3]
        assert result["property_0"].to_list() == ["value_0", "value_0"]
        assert result["agent_id"].to_list() == [0, 1]

    def test_dimensions(self, grid_moore: GridPandas):
        assert isinstance(grid_moore.dimensions, list)
        assert len(grid_moore.dimensions) == 2

    def test_empty_cells(self, grid_moore: GridPandas):
        result = grid_moore.empty_cells
        assert len(result) == 7
        assert isinstance(result, pd.DataFrame)
        assert result.columns.to_list() == ["dim_0", "dim_1"]

    def test_full_cells(self, grid_moore: GridPandas):
        grid_moore.set_cells([[0, 0], [1, 1]], {"capacity": 1})
        result = grid_moore.full_cells
        assert len(result) == 2
        assert isinstance(result, pd.DataFrame)
        assert result.columns.to_list() == ["dim_0", "dim_1"]
        assert (
            ((result["dim_0"] == 0) & (result["dim_1"] == 0))
            | ((result["dim_0"] == 1) & (result["dim_1"] == 1))
        ).all()

    def test_model(self, grid_moore: GridPandas, model: ModelDF):
        assert grid_moore.model == model

    def test_neighborhood_type(
        self,
        grid_moore: GridPandas,
        grid_von_neumann: GridPandas,
        grid_hexagonal: GridPandas,
    ):
        assert grid_moore.neighborhood_type == "moore"
        assert grid_von_neumann.neighborhood_type == "von_neumann"
        assert grid_hexagonal.neighborhood_type == "hexagonal"

    def test_random(self, grid_moore: GridPandas):
        assert grid_moore.random == grid_moore.model.random

    def test_remaining_capacity(self, grid_moore: GridPandas):
        assert grid_moore.remaining_capacity == (3 * 3 * 2 - 2)

    def test_torus(self, model: ModelDF, grid_moore: GridPandas):
        assert not grid_moore.torus

        grid_2 = GridPandas(model, [3, 3], torus=True)
        assert grid_2.torus
