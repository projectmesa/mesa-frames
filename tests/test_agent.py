from copy import copy, deepcopy

import pandas as pd
import pytest
import typeguard as tg
from mesa import Model
from numpy.random import Generator

from mesa_frames import (
    AgentSetPandas,
    AgentSetPolars,
    AgentsPandas,
    AgentsPolars,
    ModelDF,
    agent,
)


@tg.typechecked
class ExampleAgentSet(AgentSetPandas):
    def __init__(self, model: ModelDF):
        self.starting_wealth = pd.Series([1, 2, 3, 4], name="wealth")

    def add_wealth(self, amount: int) -> None:
        self.agents["wealth"] += amount


@pytest.fixture
def fix1_AgentSetPandas():
    model = ModelDF()
    agents = ExampleAgentSet(model)
    agents.add({"unique_id": [0, 1, 2, 3]})
    agents.agents["wealth"] = agents.starting_wealth
    agents.agents["age"] = [10, 20, 30, 40]
    return agents


@pytest.fixture
def fix2_AgentSetPandas():
    model = ModelDF()
    agents = ExampleAgentSet(model)
    agents.add({"unique_id": [10, 11, 12, 13]})
    agents.agents["wealth"] = agents.starting_wealth + 10
    agents.agents["age"] = [100, 200, 300, 400]

    return agents


class Test_AgentSetPandas:
    def test__init__(self):
        model = ModelDF()
        forbidden_model = Model()
        try:
            agents = ExampleAgentSet(forbidden_model)
        except Exception as e:
            assert type(e) == tg.TypeCheckError
        agents = ExampleAgentSet(model)
        assert agents.model == model
        assert isinstance(agents.agents, pd.DataFrame)
        assert isinstance(agents._mask, pd.Series)
        assert isinstance(agents.random, Generator)
        assert agents.starting_wealth.tolist() == [1, 2, 3, 4]

    def test__add__(self, fix1_AgentSetPandas, fix2_AgentSetPandas):
        agents = fix1_AgentSetPandas
        agents2 = fix2_AgentSetPandas

        # Test with two AgentSetPandas
        agents3 = agents + agents2
        assert agents3.agents.index.tolist() == [0, 1, 2, 3, 10, 11, 12, 13]

        # Test with an AgentSetPandas and a DataFrame
        agents3 = agents + agents2.agents
        assert agents3.agents.index.tolist() == [0, 1, 2, 3, 10, 11, 12, 13]

        # Test with an AgentSetPandas and a list
        agents3 = agents + [10, 5]  # 10 should be unique id and 5 should be wealth
        assert agents3.agents.index.tolist()[:-1] == [0, 1, 2, 3]
        assert len(agents3.agents) == 5
        assert agents3.agents.wealth.tolist() == [1, 2, 3, 4, 10]

        # Test with an AgentSetPandas and a dict
        agents3 = agents + {"unique_id": 10, "wealth": 5}
        assert agents3.agents.index.tolist() == [0, 1, 2, 3, 10]
        assert agents3.agents.wealth.tolist() == [1, 2, 3, 4, 5]

    def test__contains__(self, fix1_AgentSetPandas):
        # Test with a single value
        agents = fix1_AgentSetPandas
        assert 0 in agents
        assert 4 not in agents

    def test__copy__(self, fix1_AgentSetPandas):
        agents = fix1_AgentSetPandas
        agents2 = copy(agents)
        agents[0, "wealth"] = 5
        assert agents is not agents2
        assert agents[0, "wealth"].values == agents2[0, "wealth"].values
        assert agents2.model is agents.model

    def test__deepcopy__(self, fix1_AgentSetPandas):
        agents = fix1_AgentSetPandas
        agents2 = deepcopy(agents)
        agents[0, "wealth"] = 5
        assert agents is not agents2
        assert agents[0, "wealth"].values != agents2[0, "wealth"].values
        assert agents2.model is agents.model

    def test__getattr__(self, fix1_AgentSetPandas):
        agents = fix1_AgentSetPandas
        assert isinstance(agents.model, ModelDF)
        assert agents.wealth.tolist() == [1, 2, 3, 4]

    def test__getitem__(self, fix1_AgentSetPandas):
        agents = fix1_AgentSetPandas

        # Testing with a string
        assert agents["wealth"].tolist() == [1, 2, 3, 4]

        # Test with a tuple[MaskLike, str]
        assert agents[0, "wealth"].values == 1

        # Test with a list[str]
        assert agents[["wealth", "age"]].columns.tolist() == ["wealth", "age"]

        # Testing with a tuple[MaskLike, list[str]]
        result = agents[0, ["wealth", "age"]]
        assert result["wealth"].values.tolist() == [1]
        assert result["age"].values.tolist() == [10]

    def test__iadd__(self, fix1_AgentSetPandas, fix2_AgentSetPandas):
        agents = deepcopy(fix1_AgentSetPandas)
        agents2 = fix2_AgentSetPandas

        # Test with two AgentSetPandas
        agents += agents2
        assert agents.agents.index.tolist() == [0, 1, 2, 3, 10, 11, 12, 13]

        # Test with an AgentSetPandas and a DataFrame
        agents = deepcopy(fix1_AgentSetPandas)
        agents += agents2.agents
        assert agents.agents.index.tolist() == [0, 1, 2, 3, 10, 11, 12, 13]

        # Test with an AgentSetPandas and a list
        agents = deepcopy(fix1_AgentSetPandas)
        agents += [10, 5]
        assert agents.agents.index.tolist()[:-1] == [0, 1, 2, 3]
        assert len(agents.agents) == 5
        assert agents.agents.wealth.tolist() == [1, 2, 3, 4, 10]

        # Test with an AgentSetPandas and a dict
        agents = deepcopy(fix1_AgentSetPandas)
        agents += {"unique_id": 10, "wealth": 5}
        assert agents.agents.index.tolist() == [0, 1, 2, 3, 10]

    def test__iter__(self, fix1_AgentSetPandas):
        agents = fix1_AgentSetPandas
        for i, agent in agents:
            assert isinstance(agent, pd.Series)
            assert agent["wealth"] == i + 1

    def test__isub__(self, fix1_AgentSetPandas):
        agents = deepcopy(fix1_AgentSetPandas)

        # Test with two AgentSetPandas
        agents -= agents
        assert agents.agents.empty

        # Test with an AgentSetPandas and a DataFrame
        agents = deepcopy(fix1_AgentSetPandas)
        agents -= agents.agents
        assert agents.agents.empty

    def test__len__(self, fix1_AgentSetPandas):
        agents = fix1_AgentSetPandas
        assert len(agents) == 4

    def test__repr__(self, fix1_AgentSetPandas):
        agents = fix1_AgentSetPandas
        repr(agents)

    def test__reversed__(self, fix1_AgentSetPandas):
        agents = fix1_AgentSetPandas
        reversed_wealth = []
        for i, agent in reversed(agents):
            reversed_wealth.append(agent["wealth"])
        assert reversed_wealth == [4, 3, 2, 1]

    def test__setitem__(self, fix1_AgentSetPandas):
        agents = fix1_AgentSetPandas

        agents = deepcopy(agents)  # To test passing through a df later

        # Test with key=str, value=Any
        agents["wealth"] = 0
        assert agents.agents.wealth.tolist() == [0, 0, 0, 0]

        # Test with key=list[str], value=Any
        agents[["wealth", "age"]] = 1
        assert agents.agents.wealth.tolist() == [1, 1, 1, 1]
        assert agents.agents.age.tolist() == [1, 1, 1, 1]

        # Test with key=tuple, value=Any
        agents[0, "wealth"] = 5
        assert agents.agents.wealth.tolist() == [5, 1, 1, 1]

        # Test with key=MaskLike, value=Any
        agents[0] = [9, 99]
        assert agents.agents.loc[0, "wealth"] == 9
        assert agents.agents.loc[0, "age"] == 99

    def test__str__(self, fix1_AgentSetPandas):
        agents = fix1_AgentSetPandas
        str(agents)

    def test__sub__(self, fix1_AgentSetPandas):
        agents = fix1_AgentSetPandas
        agents2 = agents - agents
        assert agents2.agents.empty
        assert agents.agents.wealth.tolist() == [1, 2, 3, 4]

    def test_get_object(self, fix1_AgentSetPandas):
        agents = fix1_AgentSetPandas
        assert agents._get_obj(inplace=True) is agents
        assert agents._get_obj(inplace=False) is not agents

    def test_active_agents(self, fix1_AgentSetPandas):
        agents = fix1_AgentSetPandas
        agents.active_agents = [0, 1]
        assert agents.active_agents.index.to_list() == [0, 1]

    def test_add(self, fix1_AgentSetPandas, fix2_AgentSetPandas):
        agents = fix1_AgentSetPandas
        agents2 = fix2_AgentSetPandas

        # Test with self
        result = agents.add(agents2, inplace=False)
        assert result.agents.index.to_list() == [0, 1, 2, 3, 10, 11, 12, 13]

        # Test with a DataFrame
        result = agents.add(agents2.agents, inplace=False)
        assert result.agents.index.to_list() == [0, 1, 2, 3, 10, 11, 12, 13]

        # Test with a list
        result = agents.add([10, 5, 10], inplace=False)
        assert result.agents.index.to_list() == [0, 1, 2, 3, 10]
        assert result.agents.wealth.to_list() == [1, 2, 3, 4, 5]
        assert result.agents.age.to_list() == [10, 20, 30, 40, 10]

        # Test with a dict
        agents.add({"unique_id": [4, 5], "wealth": [5, 6], "age": [50, 60]})
        assert agents.agents.wealth.tolist() == [1, 2, 3, 4, 5, 6]
        assert agents.agents.index.tolist() == [0, 1, 2, 3, 4, 5]
        assert agents.agents.age.tolist() == [10, 20, 30, 40, 50, 60]

    def test_contains(self, fix1_AgentSetPandas):
        agents = fix1_AgentSetPandas

        # Test with a single value
        assert agents.contains(0)
        assert not agents.contains(4)

        # Test with a list
        assert agents.contains([0, 1]).values.tolist() == [True, True]

        # Test with a pd.DataFrame
        assert agents.contains(pd.DataFrame({"unique_id": [0, 4]})).values.tolist() == [True, False]

    def test_copy(self, fix1_AgentSetPandas):
        agents = fix1_AgentSetPandas

        # Test with deep=False
        agents2 = agents.copy(deep=False)
        agents2[0, "wealth"] = 5
        assert agents[0, "wealth"].values == agents2[0, "wealth"].values
        assert agents2.model is agents.model

        # Test with deep=True
        agents2 = fix1_AgentSetPandas.copy(deep=True)
        agents2[0, "wealth"] = 3
        assert agents[0, "wealth"].values != agents2[0, "wealth"].values
        assert agents2.model is agents.model

        # Test by skipping starting_wealth
        agents2 = agents.copy(skip=["starting_wealth"])
        with pytest.raises(KeyError) as e:
            agents2.starting_wealth
        assert "starting_wealth" in str(e)

    def test_discard(self, fix1_AgentSetPandas):
        agents = fix1_AgentSetPandas

        # Test with a single value
        result = agents.discard(0, inplace=False)
        assert result.agents.index.to_list() == [1, 2, 3]

        # Test with a list
        result = agents.discard([0, 1], inplace=False)
        assert agents.agents.index.tolist() == [2, 3]

        # Test with a pd.DataFrame
        result = agents.discard(pd.DataFrame({"unique_id": [0, 1]}), inplace=False)
        assert result.agents.index.to_list() == [2, 3]

        # Test with active_agents
        agents.active_agents = [0, 1]
        result = agents.discard("active", inplace=False)
        assert result.agents.index.to_list() == [2, 3]

    def test_do(self, fix1_AgentSetPandas):
        agents = fix1_AgentSetPandas
        agents.do("add_wealth", 1)
        assert agents.agents.wealth.tolist() == [2, 3, 4, 5]
        assert agents.do("add_wealth", 1, return_results=True) == None

    def test_get_attribute(self, fix1_AgentSetPandas):
        agents = fix1_AgentSetPandas

        # Test with a single attribute
        assert agents.get_attribute("wealth").tolist() == [1, 2, 3, 4]

        # Test with a list of attributes
        assert agents.get_attribute(["wealth", "age"]).columns.tolist() == [
            "wealth",
            "age",
        ]

        # Test with a single attribute and a mask
        selected = agents.select(agents["wealth"] > 1, inplace=False)
        assert selected.get_attribute("wealth", mask="active").tolist() == [2, 3, 4]

    def test_remove(self, fix1_AgentSetPandas):
        agents = fix1_AgentSetPandas
        agents.remove([0, 1])
        assert agents.agents.index.tolist() == [2, 3]
        with pytest.raises(KeyError) as e:
            agents.remove([1])
        assert "1" in str(e)

    def test_select(self, fix1_AgentSetPandas):
        agents: ExampleAgentSet = fix1_AgentSetPandas

        # Test with default arguments. Should select all agents
        selected = agents.select(inplace=False)
        assert selected.active_agents.wealth.tolist() == agents.agents.wealth.tolist()

        # Test with a pd.Series[bool]
        mask = pd.Series([True, False, True, True])
        selected = agents.select(mask, inplace=False)
        assert selected.active_agents.index.tolist() == [0, 2, 3]

        # Test with a ListLike
        mask = [0, 2]
        selected = agents.select(mask, inplace=False)
        assert selected.active_agents.index.tolist() == [0, 2]

        # Test with a pd.DataFrame
        mask = pd.DataFrame({"unique_id": [0, 1]})
        selected = agents.select(mask, inplace=False)
        assert selected.active_agents.index.tolist() == [0, 1]

        # Test with filter_func
        def filter_func(agentset: AgentSetPandas) -> pd.Series:
            return agentset.agents.wealth > 1

        selected = agents.select(filter_func=filter_func, inplace=False)
        assert selected.active_agents.index.tolist() == [1, 2, 3]

        # Test with n
        selected = agents.select(n=3, inplace=False)
        assert len(selected.active_agents) == 3

        # Test with n, filter_func and mask
        mask = pd.Series([True, False, True, True])
        selected = agents.select(mask, filter_func=filter_func, n=1, inplace=False)
        assert any(el in selected.active_agents.index.tolist() for el in [2, 3])

    def test_set_attribute(self, fix1_AgentSetPandas):
        agents = fix1_AgentSetPandas

        # Test with a single attribute
        agents.set_attribute("wealth", 0)
        assert agents.agents.wealth.tolist() == [0, 0, 0, 0]

        # Test with a list of attributes
        agents.set_attribute(["wealth", "age"], 1)

        # Test with a single attribute and a mask
        selected = agents.select(agents["wealth"] > 1, inplace=False)
        selected.set_attribute("wealth", 0, mask="active")
        assert selected.agents.wealth.tolist() == [1, 0, 0, 0]

        # Test with a dictionary
        agents.set_attribute({"wealth": 10, "age": 20})
        assert agents.agents.wealth.tolist() == [10, 10, 10, 10]
        assert agents.agents.age.tolist() == [20, 20, 20, 20]

    def test_shuffle(self, fix1_AgentSetPandas):
        agents = fix1_AgentSetPandas
        for _ in range(10):
            original_order = agents.agents.index.tolist()
            agents.shuffle()
            if original_order != agents.agents.index.tolist():
                return
        assert False

    def test_sort(self, fix1_AgentSetPandas):
        agents = fix1_AgentSetPandas
        agents.sort("wealth", ascending=False)
        assert agents.agents.wealth.tolist() == [4, 3, 2, 1]
