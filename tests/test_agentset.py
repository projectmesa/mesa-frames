import duckdb
import ibis as ib
import numpy as np
import pandas as pd
import pytest

from mesa_frames.concrete.agents import AgentSetDF
from mesa_frames.concrete.model import ModelDF

IDS = np.array([0, 1, 2, 3])
STARTING_WEALTH = np.array([1, 2, 3, 4])
MODEL = ModelDF()


class ExampleAgentSetDF(AgentSetDF):
    def __init__(self, model):
        super().__init__(model)
        self.starting_wealth = STARTING_WEALTH

    def add_wealth(self, amount: int) -> None:
        self.agents["wealth"] += amount

    def step(self) -> None:
        self.add_wealth(1)


class Test_AgentSetDF:
    def test__init__(self):
        agents = ExampleAgentSetDF(MODEL)
        assert agents.model == MODEL
        assert isinstance(agents.agents, ib.Table)
        assert np.array_equal(agents.starting_wealth, STARTING_WEALTH)

    def test_add(self):
        agents = ExampleAgentSetDF(MODEL)
        t = ib.memtable(
            {"unique_id": IDS, "wealth": STARTING_WEALTH, "age": [10, 20, 30, 40]}
        )

        agents.add(t)
        # We need to use pandas.testing.assert_frame_equal because ibis does not have it yet
        pd.testing.assert_frame_equal(
            agents.agents.to_pandas(), t.to_pandas(), check_dtype=False
        )
        # TODO: check after AgentSetDF refactoring
        # assert agents.model.agents.contains(agents.agents)

        # Test with ids that are not integers
        t2 = t.mutate(unique_id=["a", "b", "c", "d"])
        agents = ExampleAgentSetDF(MODEL)
        with pytest.raises(duckdb.ConversionException):
            agents.add(t2)

        # Test with table without unique_id
        t2 = t.drop("unique_id")
        agents = ExampleAgentSetDF(MODEL)
        with pytest.raises(KeyError):
            agents.add(t2)

        # Test with table with multiple unique_id
        t3 = t2.mutate(unique_id=0)
        agents = ExampleAgentSetDF(MODEL)
        with pytest.raises(ValueError):
            agents.add(t3)

        # Test with agents already in the AgentSetDF
        agents = ExampleAgentSetDF(MODEL)
        agents.add(t)
        with pytest.raises(ValueError):
            agents.add(t)

        # Try adding agents that are already in the model
        # TODO: after refactoring AgentsDF
