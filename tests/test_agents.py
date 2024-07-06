from copy import copy, deepcopy

import pandas as pd
import polars as pl
import pytest

from mesa_frames import AgentsDF, ModelDF
from mesa_frames.abstract.agents import AgentSetDF
from mesa_frames.types import MaskLike
from tests.test_agentset_pandas import (
    ExampleAgentSetPandas,
    fix1_AgentSetPandas,
    fix2_AgentSetPandas,
)
from tests.test_agentset_polars import (
    ExampleAgentSetPolars,
    fix2_AgentSetPolars,
)


# This serves otherwise ruff complains about the two fixtures not being used
def not_called():
    fix1_AgentSetPandas()
    fix2_AgentSetPandas()
    fix2_AgentSetPolars()


@pytest.fixture
def fix_AgentsDF(
    fix1_AgentSetPandas: ExampleAgentSetPandas,
    fix2_AgentSetPolars: ExampleAgentSetPolars,
) -> AgentsDF:
    model = ModelDF()
    agents = AgentsDF(model)
    agents.add([fix1_AgentSetPandas, fix2_AgentSetPolars])
    return agents


class Test_AgentsDF:
    def test___init__(self):
        model = ModelDF()
        agents = AgentsDF(model)
        assert agents.model == model
        assert isinstance(agents._agentsets, list)
        assert len(agents._agentsets) == 0
        assert isinstance(agents._ids, pl.Series)
        assert agents._ids.is_empty()
        assert agents._ids.name == "unique_id"

    def test_add(
        self,
        fix1_AgentSetPandas: ExampleAgentSetPandas,
        fix2_AgentSetPolars: ExampleAgentSetPolars,
    ):
        model = ModelDF()
        agents = AgentsDF(model)
        agentset_pandas = fix1_AgentSetPandas
        agentset_polars = fix2_AgentSetPolars

        # Test with a single AgentSetPandas
        result = agents.add(agentset_pandas, inplace=False)
        assert result._agentsets[0] is agentset_pandas
        assert result._ids.to_list() == agentset_pandas._agents.index.to_list()

        # Test with a single AgentSetPolars
        result = agents.add(agentset_polars, inplace=False)
        assert result._agentsets[0] is agentset_polars
        assert result._ids.to_list() == agentset_polars._agents["unique_id"].to_list()

        # Test with a list of AgentSetDFs
        result = agents.add([agentset_pandas, agentset_polars], inplace=True)
        assert result._agentsets[0] is agentset_pandas
        assert result._agentsets[1] is agentset_polars
        assert (
            result._ids.to_list()
            == agentset_pandas._agents.index.to_list()
            + agentset_polars._agents["unique_id"].to_list()
        )

        # Test if adding the same AgentSetDF raises ValueError
        with pytest.raises(ValueError):
            agents.add(agentset_pandas, inplace=False)

    def test_contains(
        self, fix2_AgentSetPandas: ExampleAgentSetPandas, fix_AgentsDF: AgentsDF
    ):
        agents = fix_AgentsDF
        agentset_pandas = agents._agentsets[0]

        # Test with an AgentSetDF
        assert agents.contains(agentset_pandas)

        # Test with an AgentSetDF not present
        assert not agents.contains(fix2_AgentSetPandas)

        # Test with an iterable of AgentSetDFs
        assert agents.contains([agentset_pandas, fix2_AgentSetPandas]).to_list() == [
            True,
            False,
        ]

        # Test with single id
        assert agents.contains(0)

        # Test with a list of ids
        assert agents.contains([0, 10]).to_list() == [True, False]

    def test_copy(self, fix_AgentsDF: AgentsDF):
        agents = fix_AgentsDF
        agents.test_list = [[1, 2, 3]]

        # Test with deep=False
        agents2 = agents.copy(deep=False)
        agents2.test_list[0].append(4)
        assert agents.test_list[0][-1] == agents2.test_list[0][-1]
        assert agents.model == agents2.model
        assert agents._agentsets[0] == agents2._agentsets[0]
        assert (agents._ids == agents2._ids).all()

        # Test with deep=True
        agents2 = fix_AgentsDF.copy(deep=True)
        agents2.test_list[0].append(4)
        assert agents.test_list[-1] != agents2.test_list[-1]
        assert agents.model == agents2.model
        assert agents._agentsets[0] != agents2._agentsets[0]
        assert (agents._ids == agents2._ids).all()

    def test_discard(
        self, fix_AgentsDF: AgentsDF, fix2_AgentSetPandas: ExampleAgentSetPandas
    ):
        agents = fix_AgentsDF
        # Test with a single AgentSetDF
        agentset_polars = agents._agentsets[1]
        result = agents.discard(agents._agentsets[0], inplace=False)
        assert isinstance(result._agentsets[0], ExampleAgentSetPolars)
        assert len(result._agentsets) == 1

        # Test with a list of AgentSetDFs
        result = agents.discard(agents._agentsets.copy(), inplace=False)
        assert len(result._agentsets) == 0

        # Test with IDs
        ids = [
            agents._agentsets[0]._agents.index[0],
            agents._agentsets[1]._agents["unique_id"][0],
        ]
        agentset_pandas = agents._agentsets[0]
        agentset_polars = agents._agentsets[1]
        result = agents.discard(ids, inplace=False)
        assert result._agentsets[0].index[0] == agentset_pandas._agents.index[1]
        assert (
            result._agentsets[1].agents["unique_id"][0]
            == agentset_polars._agents["unique_id"][1]
        )

        # Test if removing an AgentSetDF not present raises ValueError
        result = agents.discard(fix2_AgentSetPandas, inplace=False)

        # Test if removing an ID not present raises KeyError
        assert -100 not in agents._ids
        result = agents.discard(-100, inplace=False)

    def test_do(self, fix_AgentsDF: AgentsDF):
        agents = fix_AgentsDF

        expected_result_0 = agents._agentsets[0].agents["wealth"]
        expected_result_0 += 1

        expected_result_1 = agents._agentsets[1].agents["wealth"]
        expected_result_1 += 1

        # Test with no return_results, no mask, inplace
        agents.do("add_wealth", 1)
        assert (
            agents._agentsets[0].agents["wealth"].to_list()
            == expected_result_0.to_list()
        )
        assert (
            agents._agentsets[1].agents["wealth"].to_list()
            == expected_result_1.to_list()
        )

        # Test with return_results=True, no mask, inplace
        expected_result_0 = agents._agentsets[0].agents["wealth"]
        expected_result_0 += 1

        expected_result_1 = agents._agentsets[1].agents["wealth"]
        expected_result_1 += 1
        assert agents.do("add_wealth", 1, return_results=True) == {
            agents._agentsets[0]: None,
            agents._agentsets[1]: None,
        }
        assert (
            agents._agentsets[0].agents["wealth"].to_list()
            == expected_result_0.to_list()
        )
        assert (
            agents._agentsets[1].agents["wealth"].to_list()
            == expected_result_1.to_list()
        )

        # Test with a mask, inplace
        mask0 = (
            agents._agentsets[0].agents["wealth"] > 10
        )  # No agent should be selected
        mask1 = (
            agents._agentsets[1].agents["wealth"] > 10
        )  # All agents should be selected
        mask_dictionary = {agents._agentsets[0]: mask0, agents._agentsets[1]: mask1}

        expected_result_0 = agents._agentsets[0].agents["wealth"]
        expected_result_1 = agents._agentsets[1].agents["wealth"]
        expected_result_1 += 1

        agents.do("add_wealth", 1, mask=mask_dictionary)
        assert (
            agents._agentsets[0].agents["wealth"].to_list()
            == expected_result_0.to_list()
        )
        assert (
            agents._agentsets[1].agents["wealth"].to_list()
            == expected_result_1.to_list()
        )

    def test_get(
        self,
        fix_AgentsDF: AgentsDF,
        fix1_AgentSetPandas: ExampleAgentSetPandas,
        fix2_AgentSetPolars: ExampleAgentSetPolars,
    ):
        agents = fix_AgentsDF

        # Test with a single attribute
        assert (
            agents.get("wealth")[fix1_AgentSetPandas].to_list()
            == fix1_AgentSetPandas._agents["wealth"].to_list()
        )
        assert (
            agents.get("wealth")[fix2_AgentSetPolars].to_list()
            == fix2_AgentSetPolars._agents["wealth"].to_list()
        )

        # Test with a list of attributes
        result = agents.get(["wealth", "age"])
        assert result[fix1_AgentSetPandas].columns.to_list() == ["wealth", "age"]
        assert (
            result[fix1_AgentSetPandas]["wealth"].to_list()
            == fix1_AgentSetPandas._agents["wealth"].to_list()
        )
        assert (
            result[fix1_AgentSetPandas]["age"].to_list()
            == fix1_AgentSetPandas._agents["age"].to_list()
        )
        assert result[fix2_AgentSetPolars].columns == ["wealth", "age"]
        assert (
            result[fix2_AgentSetPolars]["wealth"].to_list()
            == fix2_AgentSetPolars._agents["wealth"].to_list()
        )
        assert (
            result[fix2_AgentSetPolars]["age"].to_list()
            == fix2_AgentSetPolars._agents["age"].to_list()
        )

        # Test with a single attribute and a mask
        mask0 = (
            fix1_AgentSetPandas._agents["wealth"]
            > fix1_AgentSetPandas._agents["wealth"][0]
        )
        mask1 = (
            fix2_AgentSetPolars._agents["wealth"]
            > fix2_AgentSetPolars._agents["wealth"][0]
        )
        mask_dictionary = {fix1_AgentSetPandas: mask0, fix2_AgentSetPolars: mask1}
        result = agents.get("wealth", mask=mask_dictionary)
        assert (
            result[fix1_AgentSetPandas].to_list()
            == fix1_AgentSetPandas._agents["wealth"].to_list()[1:]
        )
        assert (
            result[fix2_AgentSetPolars].to_list()
            == fix2_AgentSetPolars._agents["wealth"].to_list()[1:]
        )

    def test_remove(
        self,
        fix_AgentsDF: AgentsDF,
        fix2_AgentSetPandas: ExampleAgentSetPandas,
    ):
        agents = fix_AgentsDF

        # Test with a single AgentSetDF
        agentset_polars = agents._agentsets[1]
        result = agents.remove(agents._agentsets[0], inplace=False)
        assert isinstance(result._agentsets[0], ExampleAgentSetPolars)
        assert len(result._agentsets) == 1

        # Test with a list of AgentSetDFs
        result = agents.remove(agents._agentsets.copy(), inplace=False)
        assert len(result._agentsets) == 0

        # Test with IDs
        ids = [
            agents._agentsets[0]._agents.index[0],
            agents._agentsets[1]._agents["unique_id"][0],
        ]
        agentset_pandas = agents._agentsets[0]
        agentset_polars = agents._agentsets[1]
        result = agents.remove(ids, inplace=False)
        assert result._agentsets[0].index[0] == agentset_pandas._agents.index[1]
        assert (
            result._agentsets[1].agents["unique_id"][0]
            == agentset_polars._agents["unique_id"][1]
        )

        # Test if removing an AgentSetDF not present raises ValueError
        with pytest.raises(ValueError):
            result = agents.remove(fix2_AgentSetPandas, inplace=False)

        # Test if removing an ID not present raises KeyError
        assert -100 not in agents._ids
        with pytest.raises(KeyError):
            result = agents.remove(-100, inplace=False)

    def test_select(self, fix_AgentsDF: AgentsDF):
        agents = fix_AgentsDF

        # Test with default arguments. Should select all agents
        selected = agents.select(inplace=False)
        active_agents_dict = selected.active_agents
        agents_dict = selected.agents
        assert active_agents_dict.keys() == agents_dict.keys()
        # Using assert to compare all DataFrames in the dictionaries
        assert (
            (list(active_agents_dict.values())[0] == list(agents_dict.values())[0])
            .all()
            .all()
        )
        assert all(
            series.all()
            for series in (
                list(active_agents_dict.values())[1] == list(agents_dict.values())[1]
            )
        )

        # Test with a mask
        mask0 = pd.Series(
            [True, False, True, True], index=agents._agentsets[0].index, dtype=bool
        )
        mask1 = pl.Series("mask", [True, False, True, True], dtype=pl.Boolean)
        mask_dictionary = {agents._agentsets[0]: mask0, agents._agentsets[1]: mask1}
        selected = agents.select(mask_dictionary, inplace=False)
        assert (
            selected.active_agents[selected._agentsets[0]]["wealth"].to_list()[0]
            == agents._agentsets[0]["wealth"].to_list()[0]
        )
        assert (
            selected.active_agents[selected._agentsets[0]]["wealth"].to_list()[-1]
            == agents._agentsets[0]["wealth"].to_list()[-1]
        )
        assert (
            selected.active_agents[selected._agentsets[1]]["wealth"].to_list()[0]
            == agents._agentsets[1]["wealth"].to_list()[0]
        )
        assert (
            selected.active_agents[selected._agentsets[1]]["wealth"].to_list()[-1]
            == agents._agentsets[1]["wealth"].to_list()[-1]
        )

        # Test with filter_func
        def filter_func(agentset: AgentSetDF) -> pl.Series:
            return agentset.agents["wealth"] > agentset.agents["wealth"][0]

        selected = agents.select(filter_func=filter_func, inplace=False)
        assert (
            selected.active_agents[selected._agentsets[0]]["wealth"].to_list()
            == agents._agentsets[0]["wealth"].to_list()[1:]
        )
        assert (
            selected.active_agents[selected._agentsets[1]]["wealth"].to_list()
            == agents._agentsets[1]["wealth"].to_list()[1:]
        )

        # Test with n
        selected = agents.select(n=3, inplace=False)
        assert sum(len(df) for df in selected.active_agents.values()) in [2, 3]

        # Test with n, filter_func and mask
        selected = agents.select(
            mask_dictionary, filter_func=filter_func, n=2, inplace=False
        )
        assert any(
            el in selected.active_agents[selected._agentsets[0]]["wealth"].to_list()
            for el in agents.active_agents[agents._agentsets[0]]["wealth"].to_list()[
                2:4
            ]
        )
        assert any(
            el in selected.active_agents[selected._agentsets[1]]["wealth"].to_list()
            for el in agents.active_agents[agents._agentsets[1]]["wealth"].to_list()[
                2:4
            ]
        )

    def test_set(self, fix_AgentsDF: AgentsDF):
        agents = fix_AgentsDF

        # Test with a single attribute
        result = agents.set("wealth", 0, inplace=False)
        assert result._agentsets[0].agents["wealth"].to_list() == [0] * len(
            agents._agentsets[0]
        )
        assert result._agentsets[1].agents["wealth"].to_list() == [0] * len(
            agents._agentsets[1]
        )

        # Test with a list of attributes
        agents.set(["wealth", "age"], 1, inplace=True)
        assert agents._agentsets[0].agents["wealth"].to_list() == [1] * len(
            agents._agentsets[0]
        )
        assert agents._agentsets[0].agents["age"].to_list() == [1] * len(
            agents._agentsets[0]
        )

        # Test with a single attribute and a mask
        mask0 = pd.Series(
            [True] + [False] * (len(agents._agentsets[0]) - 1),
            index=agents._agentsets[0].index,
            dtype=bool,
        )
        mask1 = pl.Series(
            "mask", [True] + [False] * (len(agents._agentsets[1]) - 1), dtype=pl.Boolean
        )
        mask_dictionary = {agents._agentsets[0]: mask0, agents._agentsets[1]: mask1}
        result = agents.set("wealth", 0, mask=mask_dictionary, inplace=False)
        assert result._agentsets[0].agents["wealth"].to_list() == [0] + [1] * (
            len(agents._agentsets[0]) - 1
        )
        assert result._agentsets[1].agents["wealth"].to_list() == [0] + [1] * (
            len(agents._agentsets[1]) - 1
        )

        # Test with a dictionary
        agents.set(
            {agents._agentsets[0]: {"wealth": 0}, agents._agentsets[1]: {"wealth": 1}},
            inplace=True,
        )
        assert agents._agentsets[0].agents["wealth"].to_list() == [0] * len(
            agents._agentsets[0]
        )
        assert agents._agentsets[1].agents["wealth"].to_list() == [1] * len(
            agents._agentsets[1]
        )

    def test_shuffle(self, fix_AgentsDF: AgentsDF):
        agents = fix_AgentsDF
        for _ in range(100):
            original_order_0 = agents._agentsets[0].agents.index.to_list()
            original_order_1 = agents._agentsets[1].agents["unique_id"].to_list()
            agents.shuffle(inplace=True)
            if (
                original_order_0 != agents._agentsets[0].agents.index.to_list()
                and original_order_1
                != agents._agentsets[1].agents["unique_id"].to_list()
            ):
                return
        assert False

    def test_sort(self, fix_AgentsDF: AgentsDF):
        agents = fix_AgentsDF
        agents.sort("wealth", ascending=False, inplace=True)
        assert pl.Series(agents._agentsets[0].agents["wealth"]).is_sorted(
            descending=True
        )
        assert pl.Series(agents._agentsets[1].agents["wealth"]).is_sorted(
            descending=True
        )

    def test__check_ids_presence(
        self,
        fix_AgentsDF: AgentsDF,
        fix1_AgentSetPandas: ExampleAgentSetPandas,
    ):
        agents = fix_AgentsDF
        agents_different_index = deepcopy(fix1_AgentSetPandas)
        agents_different_index._agents.index = [-100, -200, -300, -400]
        result = agents._check_ids_presence([fix1_AgentSetPandas])
        assert result.filter(
            pl.col("unique_id").is_in(fix1_AgentSetPandas._agents.index)
        )["present"].all()
        assert not result.filter(
            pl.col("unique_id").is_in(agents_different_index._agents.index)
        )["present"].any()

    def test__check_agentsets_presence(
        self,
        fix_AgentsDF: AgentsDF,
        fix1_AgentSetPandas: ExampleAgentSetPandas,
        fix2_AgentSetPandas: ExampleAgentSetPandas,
    ):
        agents = fix_AgentsDF
        result = agents._check_agentsets_presence(
            [fix1_AgentSetPandas, fix2_AgentSetPandas]
        )
        assert result[0]
        assert not result[1]

    def test__get_bool_masks(self, fix_AgentsDF: AgentsDF):
        agents = fix_AgentsDF
        # Test with mask = None
        result = agents._get_bool_masks(mask=None)
        truth_value = True
        for i, mask in enumerate(result.values()):
            if isinstance(mask, pl.Expr):
                mask = agents._agentsets[i]._agents.select(mask).to_series()
            truth_value &= mask.all()
        assert truth_value

        # Test with mask = "all"
        result = agents._get_bool_masks(mask="all")
        truth_value = True
        for i, mask in enumerate(result.values()):
            if isinstance(mask, pl.Expr):
                mask = agents._agentsets[i]._agents.select(mask).to_series()
            truth_value &= mask.all()
        assert truth_value

        # Test with mask = "active"
        mask0 = (
            agents._agentsets[0].agents["wealth"]
            > agents._agentsets[0].agents["wealth"][0]
        )
        mask1 = (
            agents._agentsets[1].agents["wealth"]
            > agents._agentsets[1].agents["wealth"][0]
        )
        mask_dictionary = {agents._agentsets[0]: mask0, agents._agentsets[1]: mask1}
        agents.select(mask=mask_dictionary)
        result = agents._get_bool_masks(mask="active")
        assert result[agents._agentsets[0]].to_list() == mask0.to_list()
        assert result[agents._agentsets[1]].to_list() == mask1.to_list()

        # Test with mask = IdsLike
        result = agents._get_bool_masks(
            mask=[
                agents._agentsets[0].index[0],
                agents._agentsets[1].agents["unique_id"][0],
            ]
        )
        assert result[agents._agentsets[0]].to_list() == [True] + [False] * (
            len(agents._agentsets[0]) - 1
        )
        assert result[agents._agentsets[1]].to_list() == [True] + [False] * (
            len(agents._agentsets[1]) - 1
        )

        # Test with mask = dict[AgentSetDF, MaskLike]
        result = agents._get_bool_masks(mask=mask_dictionary)
        assert result[agents._agentsets[0]].to_list() == mask0.to_list()
        assert result[agents._agentsets[1]].to_list() == mask1.to_list()

    def test__get_obj(self, fix_AgentsDF: AgentsDF):
        agents = fix_AgentsDF
        assert agents._get_obj(inplace=True) is agents
        assert agents._get_obj(inplace=False) is not agents

    def test__return_agentsets_list(
        self,
        fix_AgentsDF: AgentsDF,
        fix1_AgentSetPandas: ExampleAgentSetPandas,
        fix2_AgentSetPandas: ExampleAgentSetPandas,
    ):
        agents = fix_AgentsDF
        result = agents._return_agentsets_list(fix1_AgentSetPandas)
        assert result == [fix1_AgentSetPandas]
        result = agents._return_agentsets_list(
            [fix1_AgentSetPandas, fix2_AgentSetPandas]
        )
        assert result == [fix1_AgentSetPandas, fix2_AgentSetPandas]

    def test___add__(
        self,
        fix1_AgentSetPandas: ExampleAgentSetPandas,
        fix2_AgentSetPolars: ExampleAgentSetPolars,
    ):
        model = ModelDF()
        agents = AgentsDF(model)
        agentset_pandas = fix1_AgentSetPandas
        agentset_polars = fix2_AgentSetPolars

        # Test with a single AgentSetPandas
        result = agents + agentset_pandas
        assert result._agentsets[0] is agentset_pandas
        assert result._ids.to_list() == agentset_pandas._agents.index.to_list()

        # Test with a single AgentSetPolars
        result = agents + agentset_polars
        assert result._agentsets[0] is agentset_polars
        assert result._ids.to_list() == agentset_polars._agents["unique_id"].to_list()

        # Test with a list of AgentSetDFs
        result = agents + [agentset_pandas, agentset_polars]
        assert result._agentsets[0] is agentset_pandas
        assert result._agentsets[1] is agentset_polars
        assert (
            result._ids.to_list()
            == agentset_pandas._agents.index.to_list()
            + agentset_polars._agents["unique_id"].to_list()
        )

        # Test if adding the same AgentSetDF raises ValueError
        with pytest.raises(ValueError):
            result + agentset_pandas

    def test___contains__(
        self, fix_AgentsDF: AgentsDF, fix2_AgentSetPandas: ExampleAgentSetPandas
    ):
        # Test with a single value
        agents = fix_AgentsDF
        agentset_pandas = agents._agentsets[0]

        # Test with an AgentSetDF
        assert agentset_pandas in agents
        # Test with an AgentSetDF not present
        assert fix2_AgentSetPandas not in agents

        # Test with single id present
        assert 0 in agents

        # Test with single id not present
        assert 10 not in agents

    def test___copy__(self, fix_AgentsDF: AgentsDF):
        agents = fix_AgentsDF
        agents.test_list = [[1, 2, 3]]

        # Test with deep=False
        agents2 = copy(agents)
        agents2.test_list[0].append(4)
        assert agents.test_list[0][-1] == agents2.test_list[0][-1]
        assert agents.model == agents2.model
        assert agents._agentsets[0] == agents2._agentsets[0]
        assert (agents._ids == agents2._ids).all()

    def test___deepcopy__(self, fix_AgentsDF: AgentsDF):
        agents = fix_AgentsDF
        agents.test_list = [[1, 2, 3]]

        agents2 = deepcopy(agents)
        agents2.test_list[0].append(4)
        assert agents.test_list[-1] != agents2.test_list[-1]
        assert agents.model == agents2.model
        assert agents._agentsets[0] != agents2._agentsets[0]
        assert (agents._ids == agents2._ids).all()

    def test___getattr__(self, fix_AgentsDF: AgentsDF):
        agents = fix_AgentsDF
        assert isinstance(agents.model, ModelDF)
        result = agents.wealth
        assert (
            result[agents._agentsets[0]].to_list()
            == agents._agentsets[0].agents["wealth"].to_list()
        )
        assert (
            result[agents._agentsets[1]].to_list()
            == agents._agentsets[1].agents["wealth"].to_list()
        )

    def test___getitem__(
        self,
        fix_AgentsDF: AgentsDF,
        fix1_AgentSetPandas: ExampleAgentSetPandas,
        fix2_AgentSetPolars: ExampleAgentSetPolars,
    ):
        agents = fix_AgentsDF

        # Test with a single attribute
        assert (
            agents["wealth"][fix1_AgentSetPandas].to_list()
            == fix1_AgentSetPandas._agents["wealth"].to_list()
        )
        assert (
            agents["wealth"][fix2_AgentSetPolars].to_list()
            == fix2_AgentSetPolars._agents["wealth"].to_list()
        )

        # Test with a list of attributes
        result = agents[["wealth", "age"]]
        assert result[fix1_AgentSetPandas].columns.to_list() == ["wealth", "age"]
        assert (
            result[fix1_AgentSetPandas]["wealth"].to_list()
            == fix1_AgentSetPandas._agents["wealth"].to_list()
        )
        assert (
            result[fix1_AgentSetPandas]["age"].to_list()
            == fix1_AgentSetPandas._agents["age"].to_list()
        )
        assert result[fix2_AgentSetPolars].columns == ["wealth", "age"]
        assert (
            result[fix2_AgentSetPolars]["wealth"].to_list()
            == fix2_AgentSetPolars._agents["wealth"].to_list()
        )
        assert (
            result[fix2_AgentSetPolars]["age"].to_list()
            == fix2_AgentSetPolars._agents["age"].to_list()
        )

        # Test with a single attribute and a mask
        mask0 = (
            fix1_AgentSetPandas._agents["wealth"]
            > fix1_AgentSetPandas._agents["wealth"][0]
        )
        mask1 = (
            fix2_AgentSetPolars._agents["wealth"]
            > fix2_AgentSetPolars._agents["wealth"][0]
        )
        mask_dictionary: dict[AgentSetDF, MaskLike] = {
            fix1_AgentSetPandas: mask0,
            fix2_AgentSetPolars: mask1,
        }
        result = agents[mask_dictionary, "wealth"]
        assert (
            result[fix1_AgentSetPandas].to_list()
            == fix1_AgentSetPandas.agents["wealth"].to_list()[1:]
        )
        assert (
            result[fix2_AgentSetPolars].to_list()
            == fix2_AgentSetPolars.agents["wealth"].to_list()[1:]
        )

    def test___iadd__(
        self,
        fix1_AgentSetPandas: ExampleAgentSetPandas,
        fix2_AgentSetPolars: ExampleAgentSetPolars,
    ):
        model = ModelDF()
        agents = AgentsDF(model)
        agentset_pandas = fix1_AgentSetPandas
        agentset_polars = fix2_AgentSetPolars

        # Test with a single AgentSetPandas
        agents_copy = deepcopy(agents)
        agents_copy += agentset_pandas
        assert agents_copy._agentsets[0] is agentset_pandas
        assert agents_copy._ids.to_list() == agentset_pandas._agents.index.to_list()

        # Test with a single AgentSetPolars
        agents_copy = deepcopy(agents)
        agents_copy += agentset_polars
        assert agents_copy._agentsets[0] is agentset_polars
        assert (
            agents_copy._ids.to_list() == agentset_polars._agents["unique_id"].to_list()
        )

        # Test with a list of AgentSetDFs
        agents_copy = deepcopy(agents)
        agents_copy += [agentset_pandas, agentset_polars]
        assert agents_copy._agentsets[0] is agentset_pandas
        assert agents_copy._agentsets[1] is agentset_polars
        assert (
            agents_copy._ids.to_list()
            == agentset_pandas._agents.index.to_list()
            + agentset_polars._agents["unique_id"].to_list()
        )

        # Test if adding the same AgentSetDF raises ValueError
        with pytest.raises(ValueError):
            agents_copy += agentset_pandas

    def test___iter__(self, fix_AgentsDF: AgentsDF):
        agents = fix_AgentsDF
        len_agentset0 = len(agents._agentsets[0])
        len_agentset1 = len(agents._agentsets[1])
        for i, agent in enumerate(agents):
            assert isinstance(agent, dict)
            if i < len_agentset0:
                assert agent["unique_id"] == agents._agentsets[0].agents.index[i]
            else:
                assert (
                    agent["unique_id"]
                    == agents._agentsets[1].agents["unique_id"][i - len_agentset0]
                )
        assert i == len_agentset0 + len_agentset1 - 1

    def test___isub__(
        self,
        fix_AgentsDF: AgentsDF,
        fix1_AgentSetPandas: ExampleAgentSetPandas,
        fix2_AgentSetPolars: ExampleAgentSetPolars,
    ):
        # Test with an AgentSetPolars and a DataFrame
        agents = fix_AgentsDF
        agents -= fix1_AgentSetPandas
        assert agents._agentsets[0] == fix2_AgentSetPolars
        assert len(agents._agentsets) == 1

    def test___len__(
        self,
        fix_AgentsDF: AgentsDF,
        fix1_AgentSetPandas: ExampleAgentSetPandas,
        fix2_AgentSetPolars: ExampleAgentSetPolars,
    ):
        assert len(fix_AgentsDF) == len(fix1_AgentSetPandas) + len(fix2_AgentSetPolars)

    def test___repr__(self, fix_AgentsDF: AgentsDF):
        repr(fix_AgentsDF)

    def test___reversed__(self, fix2_AgentSetPolars: AgentsDF):
        agents = fix2_AgentSetPolars
        reversed_wealth = []
        for agent in reversed(list(agents)):
            reversed_wealth.append(agent["wealth"])
        assert reversed_wealth == list(reversed(agents["wealth"]))

    def test___setitem__(self, fix_AgentsDF: AgentsDF):
        agents = fix_AgentsDF

        # Test with a single attribute
        agents["wealth"] = 0
        assert agents._agentsets[0].agents["wealth"].to_list() == [0] * len(
            agents._agentsets[0]
        )
        assert agents._agentsets[1].agents["wealth"].to_list() == [0] * len(
            agents._agentsets[1]
        )

        # Test with a list of attributes
        agents[["wealth", "age"]] = 1
        assert agents._agentsets[0].agents["wealth"].to_list() == [1] * len(
            agents._agentsets[0]
        )
        assert agents._agentsets[0].agents["age"].to_list() == [1] * len(
            agents._agentsets[0]
        )

        # Test with a single attribute and a mask
        mask0 = pd.Series(
            [True] + [False] * (len(agents._agentsets[0]) - 1),
            index=agents._agentsets[0].index,
            dtype=bool,
        )
        mask1 = pl.Series(
            "mask", [True] + [False] * (len(agents._agentsets[1]) - 1), dtype=pl.Boolean
        )
        mask_dictionary = {agents._agentsets[0]: mask0, agents._agentsets[1]: mask1}
        agents[mask_dictionary, "wealth"] = 0
        assert agents._agentsets[0].agents["wealth"].to_list() == [0] + [1] * (
            len(agents._agentsets[0]) - 1
        )
        assert agents._agentsets[1].agents["wealth"].to_list() == [0] + [1] * (
            len(agents._agentsets[1]) - 1
        )

    def test___str__(self, fix_AgentsDF: AgentsDF):
        str(fix_AgentsDF)

    def test___sub__(
        self,
        fix_AgentsDF: AgentsDF,
        fix1_AgentSetPandas: ExampleAgentSetPandas,
        fix2_AgentSetPolars: ExampleAgentSetPolars,
    ):
        # Test with an AgentSetPolars and a DataFrame
        result = fix_AgentsDF - fix1_AgentSetPandas
        assert isinstance(result._agentsets[0], ExampleAgentSetPolars)
        assert len(result._agentsets) == 1

    def test_agents(
        self,
        fix_AgentsDF: AgentsDF,
        fix1_AgentSetPandas: ExampleAgentSetPandas,
        fix2_AgentSetPandas: ExampleAgentSetPandas,
        fix2_AgentSetPolars: ExampleAgentSetPolars,
    ):
        assert isinstance(fix_AgentsDF.agents, dict)
        assert len(fix_AgentsDF.agents) == 2
        assert fix_AgentsDF.agents[fix1_AgentSetPandas] is fix1_AgentSetPandas._agents
        assert fix_AgentsDF.agents[fix2_AgentSetPolars] is fix2_AgentSetPolars._agents

        # Test agents.setter
        fix_AgentsDF.agents = [fix1_AgentSetPandas, fix2_AgentSetPandas]
        assert fix_AgentsDF._agentsets[0] == fix1_AgentSetPandas
        assert fix_AgentsDF._agentsets[1] == fix2_AgentSetPandas

    def test_active_agents(self, fix_AgentsDF: AgentsDF):
        agents = fix_AgentsDF

        # Test with select
        mask0 = (
            agents._agentsets[0].agents["wealth"]
            > agents._agentsets[0].agents["wealth"][0]
        )
        mask1 = (
            agents._agentsets[1].agents["wealth"]
            > agents._agentsets[1].agents["wealth"][0]
        )
        mask_dictionary = {agents._agentsets[0]: mask0, agents._agentsets[1]: mask1}
        agents1 = agents.select(mask=mask_dictionary, inplace=False)
        result = agents1.active_agents
        assert isinstance(result, dict)
        assert isinstance(result[agents1._agentsets[0]], pd.DataFrame)
        assert isinstance(result[agents1._agentsets[1]], pl.DataFrame)
        assert (
            (result[agents1._agentsets[0]] == agents1._agentsets[0]._agents[mask0])
            .all()
            .all()
        )
        assert all(
            series.all()
            for series in (
                result[agents1._agentsets[1]]
                == agents1._agentsets[1]._agents.filter(mask1)
            )
        )

        # Test with active_agents.setter
        agents1.active_agents = mask_dictionary
        result = agents1.active_agents
        assert isinstance(result, dict)
        assert isinstance(result[agents1._agentsets[0]], pd.DataFrame)
        assert isinstance(result[agents1._agentsets[1]], pl.DataFrame)
        assert (
            (result[agents1._agentsets[0]] == agents1._agentsets[0]._agents[mask0])
            .all()
            .all()
        )
        assert all(
            series.all()
            for series in (
                result[agents1._agentsets[1]]
                == agents1._agentsets[1]._agents.filter(mask1)
            )
        )

    def test_agentsets_by_type(self, fix_AgentsDF: AgentsDF):
        agents = fix_AgentsDF
        result = agents.agentsets_by_type
        assert isinstance(result, dict)
        assert isinstance(result[ExampleAgentSetPandas], AgentsDF)
        assert isinstance(result[ExampleAgentSetPolars], AgentsDF)
        assert result[ExampleAgentSetPandas]._agentsets == [agents._agentsets[0]]
        assert result[ExampleAgentSetPolars]._agentsets == [agents._agentsets[1]]

    def test_inactive_agents(self, fix_AgentsDF: AgentsDF):
        agents = fix_AgentsDF

        # Test with select
        mask0 = (
            agents._agentsets[0].agents["wealth"]
            > agents._agentsets[0].agents["wealth"][0]
        )
        mask1 = (
            agents._agentsets[1].agents["wealth"]
            > agents._agentsets[1].agents["wealth"][0]
        )
        mask_dictionary = {agents._agentsets[0]: mask0, agents._agentsets[1]: mask1}
        agents1 = agents.select(mask=mask_dictionary, inplace=False)
        result = agents1.inactive_agents
        assert isinstance(result, dict)
        assert isinstance(result[agents1._agentsets[0]], pd.DataFrame)
        assert isinstance(result[agents1._agentsets[1]], pl.DataFrame)
        assert (
            (
                result[agents1._agentsets[0]]
                == agents1._agentsets[0].select(mask0, negate=True).active_agents
            )
            .all()
            .all()
        )
        assert all(
            series.all()
            for series in (
                result[agents1._agentsets[1]]
                == agents1._agentsets[1].select(mask1, negate=True).active_agents
            )
        )
