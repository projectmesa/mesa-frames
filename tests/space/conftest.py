from __future__ import annotations

import pytest

from mesa_frames import Grid, Model
from tests.space.utils import get_unique_ids
from tests.test_agentset import ExampleAgentSet, fix1_AgentSet, fix2_AgentSet


@pytest.fixture
def model(
    fix1_AgentSet: ExampleAgentSet,
    fix2_AgentSet: ExampleAgentSet,
) -> Model:
    model = Model()
    model.sets.add([fix1_AgentSet, fix2_AgentSet])
    return model


@pytest.fixture
def grid_moore(model: Model) -> Grid:
    space = Grid(model, dimensions=[3, 3], capacity=2)
    unique_ids = get_unique_ids(model)
    space.place_agents(agents=unique_ids[[0, 1]], pos=[[0, 0], [1, 1]])
    space.cells.update([[0, 0], [1, 1]], {"capacity": [1, 3], "property_0": "value_0"})
    return space


@pytest.fixture
def grid_moore_torus(model: Model) -> Grid:
    space = Grid(model, dimensions=[3, 3], capacity=2, torus=True)
    unique_ids = get_unique_ids(model)
    space.place_agents(agents=unique_ids[[0, 1]], pos=[[0, 0], [1, 1]])
    space.cells.update([[0, 0], [1, 1]], {"capacity": [1, 3], "property_0": "value_0"})
    return space


@pytest.fixture
def grid_von_neumann(model: Model) -> Grid:
    space = Grid(model, dimensions=[3, 3], neighborhood_type="von_neumann")
    unique_ids = get_unique_ids(model)
    space.place_agents(agents=unique_ids[[0, 1]], pos=[[0, 0], [1, 1]])
    return space


@pytest.fixture
def grid_hexagonal(model: Model) -> Grid:
    space = Grid(model, dimensions=[10, 10], neighborhood_type="hexagonal")
    unique_ids = get_unique_ids(model)
    space.place_agents(agents=unique_ids[[0, 1]], pos=[[0, 0], [1, 1]])
    return space
