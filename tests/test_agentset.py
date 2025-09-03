from copy import copy, deepcopy

import polars as pl
import pytest
from numpy.random import Generator

from mesa_frames import AgentSet, Grid, Model


class ExampleAgentSet(AgentSet):
    def __init__(self, model: Model):
        super().__init__(model)
        self.starting_wealth = pl.Series("wealth", [1, 2, 3, 4])

    def add_wealth(self, amount: int) -> None:
        self["wealth"] += amount

    def step(self) -> None:
        self.add_wealth(1)


class ExampleAgentSetNoWealth(AgentSet):
    def __init__(self, model: Model):
        super().__init__(model)
        self.starting_income = pl.Series("income", [1000, 2000, 3000, 4000])

    def add_income(self, amount: int) -> None:
        self["income"] += amount

    def step(self) -> None:
        self.add_income(100)


@pytest.fixture
def fix1_AgentSet() -> ExampleAgentSet:
    model = Model()
    agents = ExampleAgentSet(model)
    agents["wealth"] = agents.starting_wealth
    agents["age"] = [10, 20, 30, 40]
    model.sets.add(agents)
    return agents


@pytest.fixture
def fix2_AgentSet() -> ExampleAgentSet:
    model = Model()
    agents = ExampleAgentSet(model)
    agents["wealth"] = agents.starting_wealth + 10
    agents["age"] = [100, 200, 300, 400]

    model.sets.add(agents)
    space = Grid(model, dimensions=[3, 3], capacity=2)
    model.space = space
    space.place_agents(agents=agents["unique_id"][[0, 1]], pos=[[2, 1], [1, 2]])
    return agents


@pytest.fixture
def fix3_AgentSet() -> ExampleAgentSet:
    model = Model()
    agents = ExampleAgentSet(model)
    agents["wealth"] = agents.starting_wealth + 7
    agents["age"] = [12, 13, 14, 116]
    return agents


@pytest.fixture
def fix1_AgentSet_with_pos(
    fix1_AgentSet: ExampleAgentSet,
) -> ExampleAgentSet:
    space = Grid(fix1_AgentSet.model, dimensions=[3, 3], capacity=2)
    fix1_AgentSet.model.space = space
    space.place_agents(agents=fix1_AgentSet["unique_id"][[0, 1]], pos=[[0, 0], [1, 1]])
    return fix1_AgentSet


@pytest.fixture
def fix1_AgentSet_no_wealth() -> ExampleAgentSetNoWealth:
    model = Model()
    agents = ExampleAgentSetNoWealth(model)
    agents["income"] = agents.starting_income
    agents["age"] = [1, 2, 3, 4]
    model.sets.add(agents)
    return agents


class Test_AgentSet:
    def test__init__(self):
        model = Model()
        agents = ExampleAgentSet(model)
        agents.add({"age": [0, 1, 2, 3]})
        assert agents.model == model
        assert isinstance(agents.df, pl.DataFrame)
        assert agents.df["age"].to_list() == [0, 1, 2, 3]
        assert isinstance(agents._mask, pl.Series)
        assert isinstance(agents.random, Generator)
        assert agents.starting_wealth.to_list() == [1, 2, 3, 4]

    def test_add(
        self,
        fix1_AgentSet: ExampleAgentSet,
    ):
        agents = fix1_AgentSet

        # Test with a pl.Dataframe
        result = agents.add(
            pl.DataFrame({"wealth": [5, 6], "age": [50, 60]}), inplace=False
        )
        assert result.df["wealth"].to_list() == [1, 2, 3, 4, 5, 6]
        assert result.df["age"].to_list() == [10, 20, 30, 40, 50, 60]

        # Test with a list (Sequence[Any])
        result = agents.add([5, 10], inplace=False)
        assert result.df["wealth"].to_list() == [1, 2, 3, 4, 5]
        assert result.df["age"].to_list() == [10, 20, 30, 40, 10]

        # Test with a dict[str, Any]
        agents.add({"wealth": [5, 6], "age": [50, 60]})
        assert agents.df["wealth"].to_list() == [1, 2, 3, 4, 5, 6]
        assert agents.df["age"].to_list() == [10, 20, 30, 40, 50, 60]

        # Test ValueError for dictionary with unique_id key (Line 131)
        with pytest.raises(
            ValueError, match="Dictionary should not have a unique_id key"
        ):
            agents.add({"wealth": [7], "age": [70], "unique_id": [999]})

        # Test ValueError for sequence length mismatch (Line 138)
        with pytest.raises(
            ValueError, match="Length of data .* must match the number of columns"
        ):
            agents.add([10])  # Only one value but agents has 2 columns (wealth, age)

        # Test with wrong sequence length
        with pytest.raises(
            ValueError, match="Length of data .* must match the number of columns"
        ):
            agents.add([10, 20, 30])  # Three values but agents has 2 columns

        # Test adding sequence to empty AgentSet - should raise ValueError
        empty_agents = ExampleAgentSet(agents.model)
        with pytest.raises(
            ValueError, match="Cannot add a sequence to an empty AgentSet"
        ):
            empty_agents.add([1, 2])  # Should raise error for empty AgentSet

    def test_contains(self, fix1_AgentSet: ExampleAgentSet):
        agents = fix1_AgentSet

        # Test with a single value
        assert agents.contains(agents["unique_id"][0])
        assert not agents.contains(0)

        # Test with a list
        assert all(agents.contains(agents["unique_id"][0, 1]) == [True, True])

        # Test with Collection (not string) - Line 177
        unique_ids = agents["unique_id"].to_list()
        result = agents.contains(unique_ids[:2])
        assert all(result == [True, True])

    def test_copy(self, fix1_AgentSet: ExampleAgentSet):
        agents = fix1_AgentSet
        agents.test_list = [[1, 2, 3]]

        # Test with deep=False
        agents2 = agents.copy(deep=False)
        agents2.test_list[0].append(4)
        assert agents.test_list[0][-1] == agents2.test_list[0][-1]

        # Test with deep=True
        agents2 = fix1_AgentSet.copy(deep=True)
        agents2.test_list[0].append(4)
        assert agents.test_list[-1] != agents2.test_list[-1]

    def test_discard(self, fix1_AgentSet_with_pos: ExampleAgentSet):
        agents = fix1_AgentSet_with_pos

        # Test with a single value
        result = agents.discard(agents["unique_id"][0], inplace=False)
        assert all(result["unique_id"] == agents["unique_id"][1, 2, 3])
        assert all(result.pos["unique_id"] == agents["unique_id"][1, 2, 3])
        assert result.pos["dim_0"].to_list() == [1, None, None]
        assert result.pos["dim_1"].to_list() == [1, None, None]
        result += pl.DataFrame({"wealth": 1, "age": 10})

        # Test with a list
        result = agents.discard(agents["unique_id"][0, 1], inplace=False)
        assert all(result["unique_id"] == agents["unique_id"][2, 3])
        assert all(result.pos["unique_id"] == agents["unique_id"][2, 3])
        assert result.pos["dim_0"].to_list() == [None, None]
        assert result.pos["dim_1"].to_list() == [None, None]

        # Test with a pl.DataFrame
        result = agents.discard(
            pl.DataFrame({"unique_id": agents["unique_id"][0, 1]}), inplace=False
        )
        assert all(result["unique_id"] == agents["unique_id"][2, 3])
        assert all(result.pos["unique_id"] == agents["unique_id"][2, 3])
        assert result.pos["dim_0"].to_list() == [None, None]
        assert result.pos["dim_1"].to_list() == [None, None]

        # Test with active_agents
        agents.active_agents = agents["unique_id"][0, 1]
        result = agents.discard("active", inplace=False)
        assert all(result["unique_id"] == agents["unique_id"][2, 3])
        assert all(result.pos["unique_id"] == agents["unique_id"][2, 3])
        assert result.pos["dim_0"].to_list() == [None, None]
        assert result.pos["dim_1"].to_list() == [None, None]

        # Test with empty list
        result = agents.discard([], inplace=False)
        assert all(result.df["unique_id"] == agents["unique_id"])

    def test_do(self, fix1_AgentSet: ExampleAgentSet):
        agents = fix1_AgentSet

        # Test with no return_results, no mask
        agents.do("add_wealth", 1)
        assert agents.df["wealth"].to_list() == [2, 3, 4, 5]

        # Test with return_results=True, no mask
        assert agents.do("add_wealth", 1, return_results=True) is None
        assert agents.df["wealth"].to_list() == [3, 4, 5, 6]

        # Test with a mask
        agents.do("add_wealth", 1, mask=agents["wealth"] > 3)
        assert agents.df["wealth"].to_list() == [3, 5, 6, 7]

    def test_get(self, fix1_AgentSet: ExampleAgentSet):
        agents = fix1_AgentSet

        # Test with a single attribute
        assert agents.get("wealth").to_list() == [1, 2, 3, 4]

        # Test with a list of attributes
        result = agents.get(["wealth", "age"])
        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["wealth", "age"]
        assert result["wealth"].to_list() == agents.df["wealth"].to_list()

        # Test with a single attribute and a mask
        selected = agents.select(agents.df["wealth"] > 1, inplace=False)
        assert selected.get("wealth", mask="active").to_list() == [2, 3, 4]

    def test_remove(self, fix1_AgentSet: ExampleAgentSet):
        agents = fix1_AgentSet
        remaining_agents_id = agents["unique_id"][2, 3]
        agents.remove(agents["unique_id"][0, 1])
        assert all(agents.df["unique_id"] == remaining_agents_id)
        with pytest.raises(KeyError):
            agents.remove([0])

    def test_select(self, fix1_AgentSet: ExampleAgentSet):
        agents = fix1_AgentSet

        # Test with default arguments. Should select all agents
        selected = agents.select(inplace=False)
        assert (
            selected.active_agents["wealth"].to_list() == agents.df["wealth"].to_list()
        )

        # Test with a pl.Series[bool]
        mask = pl.Series("mask", [True, False, True, True], dtype=pl.Boolean)
        selected = agents.select(mask, inplace=False)
        assert all(selected.active_agents["unique_id"] == agents["unique_id"][0, 2, 3])

        # Test with a ListLike
        mask = agents["unique_id"][0, 2]
        selected = agents.select(mask, inplace=False)
        assert all(selected.active_agents["unique_id"] == agents["unique_id"][0, 2])

        # Test with a pl.DataFrame
        mask = pl.DataFrame({"unique_id": agents["unique_id"][0, 1]})
        selected = agents.select(mask, inplace=False)
        assert all(selected.active_agents["unique_id"] == agents["unique_id"][0, 1])

        # Test with filter_func
        def filter_func(agentset: AgentSet) -> pl.Series:
            return agentset.df["wealth"] > 1

        selected = agents.select(filter_func=filter_func, inplace=False)
        assert all(selected.active_agents["unique_id"] == agents["unique_id"][1, 2, 3])

        # Test with n
        selected = agents.select(n=3, inplace=False)
        assert len(selected.active_agents) == 3

        # Test with n, filter_func and mask
        mask = pl.Series("mask", [True, False, True, True], dtype=pl.Boolean)
        selected = agents.select(mask, filter_func=filter_func, n=1, inplace=False)
        assert any(
            id in selected.active_agents["unique_id"].to_list()
            for id in agents["unique_id"][2, 3]
        )

    def test_set(self, fix1_AgentSet: ExampleAgentSet):
        agents = fix1_AgentSet

        # Test with a single attribute
        result = agents.set("wealth", 0, inplace=False)
        assert result.df["wealth"].to_list() == [0, 0, 0, 0]

        # Test with a list of attributes
        result = agents.set(["wealth", "age"], 1, inplace=False)
        assert result.df["wealth"].to_list() == [1, 1, 1, 1]
        assert result.df["age"].to_list() == [1, 1, 1, 1]

        # Test with a single attribute and a mask
        selected = agents.select(agents.df["wealth"] > 1, inplace=False)
        selected.set("wealth", 0, mask="active")
        assert selected.df["wealth"].to_list() == [1, 0, 0, 0]

        # Test with a dictionary
        agents.set({"wealth": 10, "age": 20})
        assert agents.df["wealth"].to_list() == [10, 10, 10, 10]
        assert agents.df["age"].to_list() == [20, 20, 20, 20]

        # Test with Collection values (Line 213) - using list as Collection
        result = agents.set("wealth", [100, 200, 300, 400], inplace=False)
        assert result.df["wealth"].to_list() == [100, 200, 300, 400]

    def test_shuffle(self, fix1_AgentSet: ExampleAgentSet):
        agents = fix1_AgentSet
        for _ in range(10):
            original_order = agents.df["unique_id"].to_list()
            agents.shuffle()
            if original_order != agents.df["unique_id"].to_list():
                return
        assert False

    def test_sort(self, fix1_AgentSet: ExampleAgentSet):
        agents = fix1_AgentSet
        agents.sort("wealth", ascending=False)
        assert agents.df["wealth"].to_list() == [4, 3, 2, 1]

    def test__add__(
        self,
        fix1_AgentSet: ExampleAgentSet,
    ):
        agents = fix1_AgentSet

        # Test with an AgentSet and a DataFrame
        agents3 = agents + pl.DataFrame({"wealth": [5, 6], "age": [50, 60]})
        assert agents3.df["wealth"].to_list() == [1, 2, 3, 4, 5, 6]
        assert agents3.df["age"].to_list() == [10, 20, 30, 40, 50, 60]

        # Test with an AgentSet and a list (Sequence[Any])
        agents3 = agents + [5, 5]  # unique_id, wealth, age
        assert all(agents3.df["unique_id"].to_list()[:-1] == agents["unique_id"])
        assert len(agents3.df) == 5
        assert agents3.df["wealth"].to_list() == [1, 2, 3, 4, 5]
        assert agents3.df["age"].to_list() == [10, 20, 30, 40, 5]

        # Test with an AgentSet and a dict
        agents3 = agents + {"age": 10, "wealth": 5}
        assert agents3.df["wealth"].to_list() == [1, 2, 3, 4, 5]

    def test__contains__(self, fix1_AgentSet: ExampleAgentSet):
        # Test with a single value
        agents = fix1_AgentSet
        assert agents["unique_id"][0] in agents
        assert 0 not in agents

    def test__copy__(self, fix1_AgentSet: ExampleAgentSet):
        agents = fix1_AgentSet
        agents.test_list = [[1, 2, 3]]

        # Test with deep=False
        agents2 = copy(agents)
        agents2.test_list[0].append(4)
        assert agents.test_list[0][-1] == agents2.test_list[0][-1]

    def test__deepcopy__(self, fix1_AgentSet: ExampleAgentSet):
        agents = fix1_AgentSet
        agents.test_list = [[1, 2, 3]]

        agents2 = deepcopy(agents)
        agents2.test_list[0].append(4)
        assert agents.test_list[-1] != agents2.test_list[-1]

    def test__getattr__(self, fix1_AgentSet: ExampleAgentSet):
        agents = fix1_AgentSet
        assert isinstance(agents.model, Model)
        assert agents.wealth.to_list() == [1, 2, 3, 4]

    def test__getitem__(self, fix1_AgentSet: ExampleAgentSet):
        agents = fix1_AgentSet

        # Testing with a string
        assert agents["wealth"].to_list() == [1, 2, 3, 4]

        # Test with a tuple[AgentMask, str]
        assert agents[agents["unique_id"][0], "wealth"].item() == 1

        # Test with a list[str]
        assert agents[["wealth", "age"]].columns == ["wealth", "age"]

        # Testing with a tuple[AgentMask, list[str]]
        result = agents[agents["unique_id"][0], ["wealth", "age"]]
        assert result["wealth"].to_list() == [1]
        assert result["age"].to_list() == [10]

    def test__iadd__(
        self,
        fix1_AgentSet: ExampleAgentSet,
    ):
        # Test with an AgentSet and a DataFrame
        agents = deepcopy(fix1_AgentSet)
        agents += pl.DataFrame({"wealth": [5, 6], "age": [50, 60]})
        assert agents.df["wealth"].to_list() == [1, 2, 3, 4, 5, 6]
        assert agents.df["age"].to_list() == [10, 20, 30, 40, 50, 60]

        # Test with an AgentSet and a list
        agents = deepcopy(fix1_AgentSet)
        agents += [5, 5]  # unique_id, wealth, age
        assert all(
            agents["unique_id"].to_list()[:-1] == fix1_AgentSet["unique_id"][0, 1, 2, 3]
        )
        assert len(agents.df) == 5
        assert agents.df["wealth"].to_list() == [1, 2, 3, 4, 5]
        assert agents.df["age"].to_list() == [10, 20, 30, 40, 5]

        # Test with an AgentSet and a dict
        agents = deepcopy(fix1_AgentSet)
        agents += {"age": 10, "wealth": 5}
        assert agents.df["wealth"].to_list() == [1, 2, 3, 4, 5]

    def test__iter__(self, fix1_AgentSet: ExampleAgentSet):
        agents = fix1_AgentSet
        for i, agent in enumerate(agents):
            assert isinstance(agent, dict)
            assert agent["wealth"] == i + 1

    def test__isub__(self, fix1_AgentSet: ExampleAgentSet):
        # Test with an AgentSet and a DataFrame
        agents = deepcopy(fix1_AgentSet)
        agents -= agents.df
        assert agents.df.is_empty()

    def test__len__(self, fix1_AgentSet: ExampleAgentSet):
        agents = fix1_AgentSet
        assert len(agents) == 4

    def test__repr__(self, fix1_AgentSet):
        agents: ExampleAgentSet = fix1_AgentSet
        repr(agents)

    def test__reversed__(self, fix1_AgentSet: ExampleAgentSet):
        agents = fix1_AgentSet
        reversed_wealth = []
        for i, agent in reversed(list(enumerate(agents))):
            reversed_wealth.append(agent["wealth"])
        assert reversed_wealth == [4, 3, 2, 1]

    def test__setitem__(self, fix1_AgentSet: ExampleAgentSet):
        agents = fix1_AgentSet

        agents = deepcopy(agents)  # To test passing through a df later

        # Test with key=str, value=Anyagents
        agents["wealth"] = 0
        assert agents.df["wealth"].to_list() == [0, 0, 0, 0]

        # Test with key=list[str], value=Any
        agents[["wealth", "age"]] = 1
        assert agents.df["wealth"].to_list() == [1, 1, 1, 1]
        assert agents.df["age"].to_list() == [1, 1, 1, 1]

        # Test with key=tuple, value=Any
        agents[agents["unique_id"][0], "wealth"] = 5
        assert agents.df["wealth"].to_list() == [5, 1, 1, 1]

        # Test with key=AgentMask, value=Any
        agents[agents["unique_id"][0]] = [9, 99]
        assert agents.df.item(0, "wealth") == 9
        assert agents.df.item(0, "age") == 99

    def test__str__(self, fix1_AgentSet: ExampleAgentSet):
        agents: ExampleAgentSet = fix1_AgentSet
        str(agents)

    def test__sub__(self, fix1_AgentSet: ExampleAgentSet):
        agents: ExampleAgentSet = fix1_AgentSet
        agents2: ExampleAgentSet = agents - agents.df
        assert agents2.df.is_empty()
        assert agents.df["wealth"].to_list() == [1, 2, 3, 4]

    def test_get_obj(self, fix1_AgentSet: ExampleAgentSet):
        agents = fix1_AgentSet
        assert agents._get_obj(inplace=True) is agents
        assert agents._get_obj(inplace=False) is not agents

    def test_agents(
        self,
        fix1_AgentSet: ExampleAgentSet,
        fix2_AgentSet: ExampleAgentSet,
    ):
        agents = fix1_AgentSet
        agents2 = fix2_AgentSet
        assert isinstance(agents.df, pl.DataFrame)

        # Test agents.setter
        agents.df = agents2.df
        assert all(agents["unique_id"] == agents2["unique_id"][0, 1, 2, 3])

    def test_active_agents(self, fix1_AgentSet: ExampleAgentSet):
        agents = fix1_AgentSet

        # Test with select
        agents.select(agents.df["wealth"] > 2, inplace=True)
        assert all(agents.active_agents["unique_id"] == agents["unique_id"][2, 3])

        # Test with active_agents.setter
        agents.active_agents = agents.df["wealth"] > 2
        assert all(agents.active_agents["unique_id"] == agents["unique_id"][2, 3])

    def test_inactive_agents(self, fix1_AgentSet: ExampleAgentSet):
        agents = fix1_AgentSet

        agents.select(agents.df["wealth"] > 2, inplace=True)
        assert all(agents.inactive_agents["unique_id"] == agents["unique_id"][0, 1])

    def test_pos(self, fix1_AgentSet_with_pos: ExampleAgentSet):
        pos = fix1_AgentSet_with_pos.pos
        assert isinstance(pos, pl.DataFrame)
        assert all(pos["unique_id"] == fix1_AgentSet_with_pos["unique_id"][0, 1, 2, 3])
        assert pos.columns == ["unique_id", "dim_0", "dim_1"]
        assert pos["dim_0"].to_list() == [0, 1, None, None]
        assert pos["dim_1"].to_list() == [0, 1, None, None]
