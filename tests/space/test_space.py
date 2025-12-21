import polars as pl
import pytest
from polars.testing import assert_frame_equal

from mesa_frames import Grid, Model
from tests.space.utils import get_unique_ids
from tests.test_agentset import ExampleAgentSet, fix1_AgentSet, fix2_AgentSet


def test_move_agents(
    grid_moore: Grid,
    fix1_AgentSet: ExampleAgentSet,
    fix2_AgentSet: ExampleAgentSet,
):
    # Test with IdsLike
    unique_ids = get_unique_ids(grid_moore.model)
    space = grid_moore.move_agents(agents=unique_ids[1], pos=[1, 1], inplace=False)
    assert space.cells.remaining_capacity == (2 * 3 * 3 - 2)
    assert_frame_equal(
        space.agents,
        pl.DataFrame(
            {"agent_id": unique_ids[[0, 1]], "dim_0": [0, 1], "dim_1": [0, 1]}
        ),
        check_row_order=False,
    )

    # Test with AgentSet
    with pytest.warns(RuntimeWarning):
        space = grid_moore.move_agents(
            agents=fix2_AgentSet,
            pos=[[0, 0], [1, 0], [2, 0], [0, 1]],
            inplace=False,
        )
    assert space.cells.remaining_capacity == (2 * 3 * 3 - 6)
    assert_frame_equal(
        space.agents,
        pl.DataFrame(
            {
                "agent_id": unique_ids[[0, 1, 4, 5, 6, 7]],
                "dim_0": [0, 1, 0, 1, 2, 0],
                "dim_1": [0, 1, 0, 0, 0, 1],
            }
        ),
        check_row_order=False,
    )

    # Test with Collection[AgentSet]
    with pytest.warns(RuntimeWarning):
        space = grid_moore.move_agents(
            agents=[fix1_AgentSet, fix2_AgentSet],
            pos=[[0, 2], [1, 2], [2, 2], [0, 1], [1, 1], [2, 1], [0, 0], [1, 0]],
            inplace=False,
        )
    assert space.cells.remaining_capacity == (2 * 3 * 3 - 8)
    assert_frame_equal(
        space.agents,
        pl.DataFrame(
            {
                "agent_id": unique_ids[[0, 1, 2, 3, 4, 5, 6, 7]],
                "dim_0": [0, 1, 2, 0, 1, 2, 0, 1],
                "dim_1": [2, 2, 2, 1, 1, 1, 0, 0],
            }
        ),
        check_row_order=False,
    )

    # Raises ValueError if len(agents) != len(pos)
    with pytest.raises(ValueError):
        grid_moore.move_agents(
            agents=unique_ids[[0, 1]], pos=[[0, 0], [1, 1], [2, 2]], inplace=False
        )

    # Test with AgentSetRegistry, pos=DataFrame
    pos = pl.DataFrame(
        {
            "dim_0": [0, 1, 2, 0, 1, 2, 0, 1],
            "dim_1": [2, 2, 2, 1, 1, 1, 0, 0],
        }
    )

    with pytest.warns(RuntimeWarning):
        space = grid_moore.move_agents(
            agents=grid_moore.model.sets,
            pos=pos,
            inplace=False,
        )
    assert space.cells.remaining_capacity == (2 * 3 * 3 - 8)
    assert_frame_equal(
        space.agents,
        pl.DataFrame(
            {
                "agent_id": unique_ids[[0, 1, 2, 3, 4, 5, 6, 7]],
                "dim_0": [0, 1, 2, 0, 1, 2, 0, 1],
                "dim_1": [2, 2, 2, 1, 1, 1, 0, 0],
            }
        ),
        check_row_order=False,
    )

    # Test with agents=int, pos=DataFrame
    pos = pl.DataFrame({"dim_0": [0], "dim_1": [2]})
    space = grid_moore.move_agents(agents=unique_ids[1], pos=pos, inplace=False)
    assert space.cells.remaining_capacity == (2 * 3 * 3 - 2)
    assert_frame_equal(
        space.agents,
        pl.DataFrame(
            {"agent_id": unique_ids[[0, 1]], "dim_0": [0, 0], "dim_1": [0, 2]}
        ),
        check_row_order=False,
    )


def test_place_agents(
    grid_moore: Grid,
    fix1_AgentSet: ExampleAgentSet,
    fix2_AgentSet: ExampleAgentSet,
):
    # Test with IdsLike
    unique_ids = get_unique_ids(grid_moore.model)
    with pytest.warns(RuntimeWarning):
        space = grid_moore.place_agents(
            agents=unique_ids[[1, 2]], pos=[[1, 1], [2, 2]], inplace=False
        )
    assert space.cells.remaining_capacity == (2 * 3 * 3 - 3)
    assert len(space.agents) == 3
    assert (
        space.agents.select(pl.col("agent_id")).to_series().to_list()
        == unique_ids[[0, 1, 2]].to_list()
    )
    assert space.agents.select(pl.col("dim_0")).to_series().to_list() == [0, 1, 2]
    assert space.agents.select(pl.col("dim_1")).to_series().to_list() == [0, 1, 2]

    # Test with agents not in the model
    with pytest.raises(ValueError):
        grid_moore.place_agents(
            agents=[0, 1],
            pos=[[0, 0], [1, 0]],
            inplace=False,
        )

    # Test with AgentSet
    space = grid_moore.place_agents(
        agents=fix2_AgentSet,
        pos=[[0, 0], [1, 0], [2, 0], [0, 1]],
        inplace=False,
    )
    unique_ids = get_unique_ids(space.model)
    assert space.cells.remaining_capacity == (2 * 3 * 3 - 6)
    assert len(space.agents) == 6
    assert_frame_equal(
        space.agents,
        pl.DataFrame(
            {
                "agent_id": unique_ids[[0, 1, 4, 5, 6, 7]],
                "dim_0": [0, 1, 0, 1, 2, 0],
                "dim_1": [0, 1, 0, 0, 0, 1],
            }
        ),
        check_row_order=False,
    )

    # Test with Collection[AgentSet]
    with pytest.warns(RuntimeWarning):
        space = grid_moore.place_agents(
            agents=[fix1_AgentSet, fix2_AgentSet],
            pos=[[0, 2], [1, 2], [2, 2], [0, 1], [1, 1], [2, 1], [0, 0], [1, 0]],
            inplace=False,
        )
    unique_ids = get_unique_ids(space.model)
    assert space.cells.remaining_capacity == (2 * 3 * 3 - 8)
    assert_frame_equal(
        space.agents,
        pl.DataFrame(
            {
                "agent_id": unique_ids[[0, 1, 2, 3, 4, 5, 6, 7]],
                "dim_0": [0, 1, 2, 0, 1, 2, 0, 1],
                "dim_1": [2, 2, 2, 1, 1, 1, 0, 0],
            }
        ),
        check_row_order=False,
    )

    # Test with AgentSetRegistry, pos=DataFrame
    pos = pl.DataFrame(
        {
            "dim_0": [0, 1, 2, 0, 1, 2, 0, 1],
            "dim_1": [2, 2, 2, 1, 1, 1, 0, 0],
        }
    )
    with pytest.warns(RuntimeWarning):
        space = grid_moore.place_agents(
            agents=grid_moore.model.sets,
            pos=pos,
            inplace=False,
        )
    assert space.cells.remaining_capacity == (2 * 3 * 3 - 8)
    assert_frame_equal(
        space.agents,
        pl.DataFrame(
            {
                "agent_id": unique_ids[[0, 1, 2, 3, 4, 5, 6, 7]],
                "dim_0": [0, 1, 2, 0, 1, 2, 0, 1],
                "dim_1": [2, 2, 2, 1, 1, 1, 0, 0],
            }
        ),
        check_row_order=False,
    )

    # Test with agents=unique_id, pos=DataFrame
    pos = pl.DataFrame({"dim_0": [0], "dim_1": [2]})
    with pytest.warns(RuntimeWarning):
        space = grid_moore.place_agents(agents=unique_ids[1], pos=pos, inplace=False)
    assert space.cells.remaining_capacity == (2 * 3 * 3 - 2)
    assert_frame_equal(
        space.agents,
        pl.DataFrame(
            {"agent_id": unique_ids[[0, 1]], "dim_0": [0, 0], "dim_1": [0, 2]}
        ),
        check_row_order=False,
    )


def test_swap_agents(
    grid_moore: Grid,
    fix1_AgentSet: ExampleAgentSet,
    fix2_AgentSet: ExampleAgentSet,
):
    unique_ids = get_unique_ids(grid_moore.model)
    grid_moore.move_agents(
        unique_ids[[0, 1, 2, 3, 4, 5, 6, 7]],
        [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]],
    )
    # Test with IdsLike
    space = grid_moore.swap_agents(unique_ids[[0, 1]], unique_ids[[2, 3]], inplace=False)
    assert (
        space.agents.filter(pl.col("agent_id") == unique_ids[0]).row(0)[1:]
        == grid_moore.agents.filter(pl.col("agent_id") == unique_ids[2]).row(0)[1:]
    )
    assert (
        space.agents.filter(pl.col("agent_id") == unique_ids[1]).row(0)[1:]
        == grid_moore.agents.filter(pl.col("agent_id") == unique_ids[3]).row(0)[1:]
    )
    assert (
        space.agents.filter(pl.col("agent_id") == unique_ids[2]).row(0)[1:]
        == grid_moore.agents.filter(pl.col("agent_id") == unique_ids[0]).row(0)[1:]
    )
    assert (
        space.agents.filter(pl.col("agent_id") == unique_ids[3]).row(0)[1:]
        == grid_moore.agents.filter(pl.col("agent_id") == unique_ids[1]).row(0)[1:]
    )
    # Test with AgentSets
    space = grid_moore.swap_agents(fix1_AgentSet, fix2_AgentSet, inplace=False)
    assert (
        space.agents.filter(pl.col("agent_id") == unique_ids[0]).row(0)[1:]
        == grid_moore.agents.filter(pl.col("agent_id") == unique_ids[4]).row(0)[1:]
    )
    assert (
        space.agents.filter(pl.col("agent_id") == unique_ids[1]).row(0)[1:]
        == grid_moore.agents.filter(pl.col("agent_id") == unique_ids[5]).row(0)[1:]
    )
    assert (
        space.agents.filter(pl.col("agent_id") == unique_ids[2]).row(0)[1:]
        == grid_moore.agents.filter(pl.col("agent_id") == unique_ids[6]).row(0)[1:]
    )
    assert (
        space.agents.filter(pl.col("agent_id") == unique_ids[3]).row(0)[1:]
        == grid_moore.agents.filter(pl.col("agent_id") == unique_ids[7]).row(0)[1:]
    )


def test_random_agents(grid_moore: Grid):
    different = False
    agents0 = grid_moore.random_agents(1)
    for _ in range(100):
        agents1 = grid_moore.random_agents(1)
        if (agents0.to_numpy() != agents1.to_numpy()).all():
            different = True
            break
    assert different


def test_random(grid_moore: Grid):
    assert grid_moore.random == grid_moore.model.random


def test_agents(grid_moore: Grid):
    unique_ids = get_unique_ids(grid_moore.model)
    assert_frame_equal(
        grid_moore.agents,
        pl.DataFrame(
            {"agent_id": unique_ids[[0, 1]], "dim_0": [0, 1], "dim_1": [0, 1]}
        ),
    )


def test_model(grid_moore: Grid, model: Model):
    assert grid_moore.model == model
