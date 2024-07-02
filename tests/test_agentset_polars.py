from copy import copy, deepcopy

import polars as pl
import pytest
import typeguard as tg
from numpy.random import Generator

from mesa_frames import AgentSetPolars, ModelDF


@tg.typechecked
class ExampleAgentSet(AgentSetPolars):
    def __init__(self, model: ModelDF):
        super().__init__(model)
        self.starting_wealth = pl.Series("wealth", [1, 2, 3, 4])

    def add_wealth(self, amount: int) -> None:
        self["wealth"] += amount


@pytest.fixture
def fix1_AgentSetPolars() -> ExampleAgentSet:
    model = ModelDF()
    agents = ExampleAgentSet(model)
    agents.add({"unique_id": [0, 1, 2, 3]})
    agents["wealth"] = agents.starting_wealth
    agents["age"] = [10, 20, 30, 40]
    return agents


@pytest.fixture
def fix2_AgentSetPolars() -> ExampleAgentSet:
    model = ModelDF()
    agents = ExampleAgentSet(model)
    agents.add({"unique_id": [4, 5, 6, 7]})
    agents["wealth"] = agents.starting_wealth + 10
    agents["age"] = [100, 200, 300, 400]
    return agents


class Test_AgentSetPolars:
    def test__init__(self):
        model = ModelDF()
        agents = ExampleAgentSet(model)
        agents.add({"unique_id": [0, 1, 2, 3]})
        assert agents.model == model
        assert isinstance(agents.agents, pl.DataFrame)
        assert agents.agents["unique_id"].to_list() == [0, 1, 2, 3]
        assert isinstance(agents._mask, pl.Series)
        assert isinstance(agents.random, Generator)
        assert agents.starting_wealth.to_list() == [1, 2, 3, 4]

    def test_add(
        self, fix1_AgentSetPolars: ExampleAgentSet, fix2_AgentSetPolars: ExampleAgentSet
    ):
        agents = fix1_AgentSetPolars
        agents2 = fix2_AgentSetPolars

        # Test with a DataFrame
        result = agents.add(agents2.agents, inplace=False)
        assert result.agents["unique_id"].to_list() == [0, 1, 2, 3, 4, 5, 6, 7]

        # Test with a list (Sequence[Any])
        result = agents.add([10, 5, 10], inplace=False)
        assert result.agents["unique_id"].to_list() == [0, 1, 2, 3, 10]
        assert result.agents["wealth"].to_list() == [1, 2, 3, 4, 5]
        assert result.agents["age"].to_list() == [10, 20, 30, 40, 10]

        # Test with a dict[str, Any]
        agents.add({"unique_id": [4, 5], "wealth": [5, 6], "age": [50, 60]})
        assert agents.agents["wealth"].to_list() == [1, 2, 3, 4, 5, 6]
        assert agents.agents["unique_id"].to_list() == [0, 1, 2, 3, 4, 5]
        assert agents.agents["age"].to_list() == [10, 20, 30, 40, 50, 60]

    def test_contains(self, fix1_AgentSetPolars: ExampleAgentSet):
        agents = fix1_AgentSetPolars

        # Test with a single value
        assert agents.contains(0)
        assert not agents.contains(4)

        # Test with a list
        assert agents.contains([0, 1]).to_list() == [True, True]

    def test_copy(self, fix1_AgentSetPolars: ExampleAgentSet):
        agents = fix1_AgentSetPolars
        agents.test_list = [[1, 2, 3]]

        # Test with deep=False
        agents2 = agents.copy(deep=False)
        agents2.test_list[0].append(4)
        assert agents.test_list[0][-1] == agents2.test_list[0][-1]

        # Test with deep=True
        agents2 = fix1_AgentSetPolars.copy(deep=True)
        agents2.test_list[0].append(4)
        assert agents.test_list[-1] != agents2.test_list[-1]

    def test_discard(self, fix1_AgentSetPolars: ExampleAgentSet):
        agents = fix1_AgentSetPolars

        # Test with a single value
        result = agents.discard(0, inplace=False)
        assert result.agents["unique_id"].to_list() == [1, 2, 3]

        # Test with a list
        result = agents.discard([0, 1], inplace=False)
        assert result.agents["unique_id"].to_list() == [2, 3]

        # Test with a pl.DataFrame
        result = agents.discard(pl.DataFrame({"unique_id": [0, 1]}), inplace=False)
        assert result.agents["unique_id"].to_list() == [2, 3]

        # Test with active_agents
        agents.active_agents = [0, 1]
        result = agents.discard("active", inplace=False)
        assert result.agents["unique_id"].to_list() == [2, 3]

    def test_do(self, fix1_AgentSetPolars: ExampleAgentSet):
        agents = fix1_AgentSetPolars

        # Test with no return_results, no mask
        agents.do("add_wealth", 1)
        assert agents.agents["wealth"].to_list() == [2, 3, 4, 5]

        # Test with return_results=True, no mask
        assert agents.do("add_wealth", 1, return_results=True) is None
        assert agents.agents["wealth"].to_list() == [3, 4, 5, 6]

        # Test with a mask
        agents.do("add_wealth", 1, mask=agents["wealth"] > 3)
        assert agents.agents["wealth"].to_list() == [3, 5, 6, 7]

    def test_get(self, fix1_AgentSetPolars: ExampleAgentSet):
        agents = fix1_AgentSetPolars

        # Test with a single attribute
        assert agents.get("wealth").to_list() == [1, 2, 3, 4]

        # Test with a list of attributes
        result = agents.get(["wealth", "age"])
        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["wealth", "age"]
        assert result["wealth"].to_list() == agents.agents["wealth"].to_list()

        # Test with a single attribute and a mask
        selected = agents.select(agents.agents["wealth"] > 1, inplace=False)
        assert selected.get("wealth", mask="active").to_list() == [2, 3, 4]

    def test_remove(self, fix1_AgentSetPolars: ExampleAgentSet):
        agents = fix1_AgentSetPolars
        agents.remove([0, 1])
        assert agents.agents["unique_id"].to_list() == [2, 3]
        with pytest.raises(KeyError):
            agents.remove([1])

    def test_select(self, fix1_AgentSetPolars: ExampleAgentSet):
        agents = fix1_AgentSetPolars

        # Test with default arguments. Should select all agents
        selected = agents.select(inplace=False)
        assert (
            selected.active_agents["wealth"].to_list()
            == agents.agents["wealth"].to_list()
        )

        # Test with a pl.Series[bool]
        mask = pl.Series("mask", [True, False, True, True], dtype=pl.Boolean)
        selected = agents.select(mask, inplace=False)
        assert selected.active_agents["unique_id"].to_list() == [0, 2, 3]

        # Test with a ListLike
        mask = [0, 2]
        selected = agents.select(mask, inplace=False)
        assert selected.active_agents["unique_id"].to_list() == [0, 2]

        # Test with a pl.DataFrame
        mask = pl.DataFrame({"unique_id": [0, 1]})
        selected = agents.select(mask, inplace=False)
        assert selected.active_agents["unique_id"].to_list() == [0, 1]

        # Test with filter_func
        def filter_func(agentset: AgentSetPolars) -> pl.Series:
            return agentset.agents["wealth"] > 1

        selected = agents.select(filter_func=filter_func, inplace=False)
        assert selected.active_agents["unique_id"].to_list() == [1, 2, 3]

        # Test with n
        selected = agents.select(n=3, inplace=False)
        assert len(selected.active_agents) == 3

        # Test with n, filter_func and mask
        mask = pl.Series("mask", [True, False, True, True], dtype=pl.Boolean)
        selected = agents.select(mask, filter_func=filter_func, n=1, inplace=False)
        assert any(el in selected.active_agents["unique_id"].to_list() for el in [2, 3])

    def test_set(self, fix1_AgentSetPolars: ExampleAgentSet):
        agents = fix1_AgentSetPolars

        # Test with a single attribute
        result = agents.set("wealth", 0, inplace=False)
        assert result.agents["wealth"].to_list() == [0, 0, 0, 0]

        # Test with a list of attributes
        result = agents.set(["wealth", "age"], 1, inplace=False)
        assert result.agents["wealth"].to_list() == [1, 1, 1, 1]
        assert result.agents["age"].to_list() == [1, 1, 1, 1]

        # Test with a single attribute and a mask
        selected = agents.select(agents.agents["wealth"] > 1, inplace=False)
        selected.set("wealth", 0, mask="active")
        assert selected.agents["wealth"].to_list() == [1, 0, 0, 0]

        # Test with a dictionary
        agents.set({"wealth": 10, "age": 20})
        assert agents.agents["wealth"].to_list() == [10, 10, 10, 10]
        assert agents.agents["age"].to_list() == [20, 20, 20, 20]

    def test_shuffle(self, fix1_AgentSetPolars: ExampleAgentSet):
        agents = fix1_AgentSetPolars
        for _ in range(10):
            original_order = agents.agents["unique_id"].to_list()
            agents.shuffle()
            if original_order != agents.agents["unique_id"].to_list():
                return
        assert False

    def test_sort(self, fix1_AgentSetPolars: ExampleAgentSet):
        agents = fix1_AgentSetPolars
        agents.sort("wealth", ascending=False)
        assert agents.agents["wealth"].to_list() == [4, 3, 2, 1]

    def test__add__(
        self, fix1_AgentSetPolars: ExampleAgentSet, fix2_AgentSetPolars: ExampleAgentSet
    ):
        agents = fix1_AgentSetPolars
        agents2 = fix2_AgentSetPolars

        # Test with an AgentSetPolars and a DataFrame
        agents3 = agents + agents2.agents
        assert agents3.agents["unique_id"].to_list() == [0, 1, 2, 3, 4, 5, 6, 7]

        # Test with an AgentSetPolars and a list (Sequence[Any])
        agents3 = agents + [10, 5, 5]  # unique_id, wealth, age
        assert agents3.agents["unique_id"].to_list()[:-1] == [0, 1, 2, 3]
        assert len(agents3.agents) == 5
        assert agents3.agents["wealth"].to_list() == [1, 2, 3, 4, 5]
        assert agents3.agents["age"].to_list() == [10, 20, 30, 40, 5]

        # Test with an AgentSetPolars and a dict
        agents3 = agents + {"unique_id": 10, "wealth": 5}
        assert agents3.agents["unique_id"].to_list() == [0, 1, 2, 3, 10]
        assert agents3.agents["wealth"].to_list() == [1, 2, 3, 4, 5]

    def test__contains__(self, fix1_AgentSetPolars: ExampleAgentSet):
        # Test with a single value
        agents = fix1_AgentSetPolars
        assert 0 in agents
        assert 4 not in agents

    def test__copy__(self, fix1_AgentSetPolars: ExampleAgentSet):
        agents = fix1_AgentSetPolars
        agents.test_list = [[1, 2, 3]]

        # Test with deep=False
        agents2 = copy(agents)
        agents2.test_list[0].append(4)
        assert agents.test_list[0][-1] == agents2.test_list[0][-1]

    def test__deepcopy__(self, fix1_AgentSetPolars: ExampleAgentSet):
        agents = fix1_AgentSetPolars
        agents.test_list = [[1, 2, 3]]

        agents2 = deepcopy(agents)
        agents2.test_list[0].append(4)
        assert agents.test_list[-1] != agents2.test_list[-1]

    def test__getattr__(self, fix1_AgentSetPolars: ExampleAgentSet):
        agents = fix1_AgentSetPolars
        assert isinstance(agents.model, ModelDF)
        assert agents.wealth.to_list() == [1, 2, 3, 4]

    def test__getitem__(self, fix1_AgentSetPolars: ExampleAgentSet):
        agents = fix1_AgentSetPolars

        # Testing with a string
        assert agents["wealth"].to_list() == [1, 2, 3, 4]

        # Test with a tuple[MaskLike, str]
        assert agents[0, "wealth"].item() == 1

        # Test with a list[str]
        assert agents[["wealth", "age"]].columns == ["wealth", "age"]

        # Testing with a tuple[MaskLike, list[str]]
        result = agents[0, ["wealth", "age"]]
        assert result["wealth"].to_list() == [1]
        assert result["age"].to_list() == [10]

    def test__iadd__(
        self, fix1_AgentSetPolars: ExampleAgentSet, fix2_AgentSetPolars: ExampleAgentSet
    ):
        agents = deepcopy(fix1_AgentSetPolars)
        agents2 = fix2_AgentSetPolars

        # Test with an AgentSetPolars and a DataFrame
        agents = deepcopy(fix1_AgentSetPolars)
        agents += agents2.agents
        assert agents.agents["unique_id"].to_list() == [0, 1, 2, 3, 4, 5, 6, 7]

        # Test with an AgentSetPolars and a list
        agents = deepcopy(fix1_AgentSetPolars)
        agents += [10, 5, 5]  # unique_id, wealth, age
        assert agents.agents["unique_id"].to_list()[:-1] == [0, 1, 2, 3]
        assert len(agents.agents) == 5
        assert agents.agents["wealth"].to_list() == [1, 2, 3, 4, 5]
        assert agents.agents["age"].to_list() == [10, 20, 30, 40, 5]

        # Test with an AgentSetPolars and a dict
        agents = deepcopy(fix1_AgentSetPolars)
        agents += {"unique_id": 10, "wealth": 5}
        assert agents.agents["unique_id"].to_list() == [0, 1, 2, 3, 10]
        assert agents.agents["wealth"].to_list() == [1, 2, 3, 4, 5]

    def test__iter__(self, fix1_AgentSetPolars: ExampleAgentSet):
        agents = fix1_AgentSetPolars
        for i, agent in enumerate(agents):
            assert isinstance(agent, dict)
            assert agent["wealth"] == i + 1

    def test__isub__(self, fix1_AgentSetPolars: ExampleAgentSet):
        # Test with an AgentSetPolars and a DataFrame
        agents = deepcopy(fix1_AgentSetPolars)
        agents -= agents.agents
        assert agents.agents.is_empty()

    def test__len__(self, fix1_AgentSetPolars: ExampleAgentSet):
        agents = fix1_AgentSetPolars
        assert len(agents) == 4

    def test__repr__(self, fix1_AgentSetPolars):
        agents: ExampleAgentSet = fix1_AgentSetPolars
        repr(agents)

    def test__reversed__(self, fix1_AgentSetPolars: ExampleAgentSet):
        agents = fix1_AgentSetPolars
        reversed_wealth = []
        for i, agent in reversed(list(enumerate(agents))):
            reversed_wealth.append(agent["wealth"])
        assert reversed_wealth == [4, 3, 2, 1]

    def test__setitem__(self, fix1_AgentSetPolars: ExampleAgentSet):
        agents = fix1_AgentSetPolars

        agents = deepcopy(agents)  # To test passing through a df later

        # Test with key=str, value=Any
        agents["wealth"] = 0
        assert agents.agents["wealth"].to_list() == [0, 0, 0, 0]

        # Test with key=list[str], value=Any
        agents[["wealth", "age"]] = 1
        assert agents.agents["wealth"].to_list() == [1, 1, 1, 1]
        assert agents.agents["age"].to_list() == [1, 1, 1, 1]

        # Test with key=tuple, value=Any
        agents[0, "wealth"] = 5
        assert agents.agents["wealth"].to_list() == [5, 1, 1, 1]

        # Test with key=MaskLike, value=Any
        agents[0] = [9, 99]
        assert agents.agents.item(0, "wealth") == 9
        assert agents.agents.item(0, "age") == 99

    def test__str__(self, fix1_AgentSetPolars: ExampleAgentSet):
        agents: ExampleAgentSet = fix1_AgentSetPolars
        str(agents)

    def test__sub__(self, fix1_AgentSetPolars: ExampleAgentSet):
        agents: ExampleAgentSet = fix1_AgentSetPolars
        agents2: ExampleAgentSet = agents - agents.agents
        assert agents2.agents.is_empty()
        assert agents.agents["wealth"].to_list() == [1, 2, 3, 4]

    def test_get_obj(self, fix1_AgentSetPolars: ExampleAgentSet):
        agents = fix1_AgentSetPolars
        assert agents._get_obj(inplace=True) is agents
        assert agents._get_obj(inplace=False) is not agents

    def test_agents(
        self, fix1_AgentSetPolars: ExampleAgentSet, fix2_AgentSetPolars: ExampleAgentSet
    ):
        agents = fix1_AgentSetPolars
        agents2 = fix2_AgentSetPolars
        assert isinstance(agents.agents, pl.DataFrame)

        # Test agents.setter
        agents.agents = agents2.agents
        assert agents.agents["unique_id"].to_list() == [4, 5, 6, 7]

    def test_active_agents(self, fix1_AgentSetPolars: ExampleAgentSet):
        agents = fix1_AgentSetPolars

        # Test with select
        agents.select(agents.agents["wealth"] > 2, inplace=True)
        assert agents.active_agents["unique_id"].to_list() == [2, 3]

        # Test with active_agents.setter
        agents.active_agents = agents.agents["wealth"] > 2
        assert agents.active_agents["unique_id"].to_list() == [2, 3]

    def test_inactive_agents(self, fix1_AgentSetPolars: ExampleAgentSet):
        agents = fix1_AgentSetPolars

        agents.select(agents.agents["wealth"] > 2, inplace=True)
        assert agents.inactive_agents["unique_id"].to_list() == [0, 1]
