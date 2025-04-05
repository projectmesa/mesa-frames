from copy import copy, deepcopy

import polars as pl
import pytest

from mesa_frames import AgentsDF, ModelDF
from mesa_frames.abstract.agents import AgentSetDF
from mesa_frames.types_ import AgentMask
from tests.test_agentset import (
    ExampleAgentSetPolars,
    fix1_AgentSetPolars,
    fix2_AgentSetPolars,
    fix3_AgentSetPolars,
)


# This serves otherwise ruff complains about the two fixtures not being used
def not_called():
    fix2_AgentSetPolars()


@pytest.fixture
def fix_AgentsDF(
    fix1_AgentSetPolars: ExampleAgentSetPolars,
    fix2_AgentSetPolars: ExampleAgentSetPolars,
) -> AgentsDF:
    model = ModelDF()
    agents = AgentsDF(model)
    agents.add([fix1_AgentSetPolars, fix2_AgentSetPolars])
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
        fix1_AgentSetPolars: ExampleAgentSetPolars,
        fix2_AgentSetPolars: ExampleAgentSetPolars,
    ):
        model = ModelDF()
        agents = AgentsDF(model)
        agentset_polars1 = fix1_AgentSetPolars
        agentset_polars2 = fix2_AgentSetPolars

        # Test with a single AgentSetPolars
        result = agents.add(agentset_polars1, inplace=False)
        assert result._agentsets[0] is agentset_polars1
        assert result._ids.to_list() == agentset_polars1._agents["unique_id"].to_list()

        # Test with a list of AgentSetDFs
        result = agents.add([agentset_polars1, agentset_polars2], inplace=True)
        assert result._agentsets[0] is agentset_polars1
        assert result._agentsets[1] is agentset_polars2
        assert (
            result._ids.to_list()
            == agentset_polars1._agents["unique_id"].to_list()
            + agentset_polars2._agents["unique_id"].to_list()
        )

        # Test if adding the same AgentSetDF raises ValueError
        with pytest.raises(ValueError):
            agents.add(agentset_polars1, inplace=False)

    def test_contains(
        self,
        fix1_AgentSetPolars: ExampleAgentSetPolars,
        fix2_AgentSetPolars: ExampleAgentSetPolars,
        fix3_AgentSetPolars: ExampleAgentSetPolars,
        fix_AgentsDF: AgentsDF,
    ):
        agents = fix_AgentsDF
        agentset_polars1 = agents._agentsets[0]

        # Test with an AgentSetDF
        assert agents.contains(agentset_polars1)
        assert agents.contains(fix1_AgentSetPolars)
        assert agents.contains(fix2_AgentSetPolars)

        # Test with an AgentSetDF not present
        assert not agents.contains(fix3_AgentSetPolars)

        # Test with an iterable of AgentSetDFs
        assert agents.contains([agentset_polars1, fix3_AgentSetPolars]).to_list() == [
            True,
            False,
        ]

        # Test with single id
        assert agents.contains(agentset_polars1["unique_id"][0])

        # Test with a list of ids
        assert agents.contains([agentset_polars1["unique_id"][0], 0]).to_list() == [
            True,
            False,
        ]

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
        self, fix_AgentsDF: AgentsDF, fix2_AgentSetPolars: ExampleAgentSetPolars
    ):
        agents = fix_AgentsDF
        # Test with a single AgentSetDF
        agentset_polars2 = agents._agentsets[1]
        result = agents.discard(agents._agentsets[0], inplace=False)
        assert isinstance(result._agentsets[0], ExampleAgentSetPolars)
        assert len(result._agentsets) == 1

        # Test with a list of AgentSetDFs
        result = agents.discard(agents._agentsets.copy(), inplace=False)
        assert len(result._agentsets) == 0

        # Test with IDs
        ids = [
            agents._agentsets[0]._agents["unique_id"][0],
            agents._agentsets[1]._agents["unique_id"][0],
        ]
        agentset_polars1 = agents._agentsets[0]
        agentset_polars2 = agents._agentsets[1]
        result = agents.discard(ids, inplace=False)
        assert (
            result._agentsets[0]["unique_id"][0]
            == agentset_polars1._agents.select("unique_id").row(1)[0]
        )
        assert (
            result._agentsets[1].agents["unique_id"][0]
            == agentset_polars2._agents["unique_id"][1]
        )

        # Test if removing an AgentSetDF not present raises ValueError
        result = agents.discard(fix2_AgentSetPolars, inplace=False)

        # Test if removing an ID not present raises KeyError
        assert 0 not in agents._ids
        result = agents.discard(0, inplace=False)

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

        # original_index_for_pandas_df = agents._agentsets[0].index.copy()
        agents.do("add_wealth", 1, mask=mask_dictionary)
        # agents._agentsets[0].agents = agents._agentsets[0].agents.reindex(
        #     original_index_for_pandas_df
        # )
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
        fix1_AgentSetPolars: ExampleAgentSetPolars,
        fix2_AgentSetPolars: ExampleAgentSetPolars,
    ):
        agents = fix_AgentsDF

        # Test with a single attribute
        assert (
            agents.get("wealth")[fix1_AgentSetPolars].to_list()
            == fix1_AgentSetPolars._agents["wealth"].to_list()
        )
        assert (
            agents.get("wealth")[fix2_AgentSetPolars].to_list()
            == fix2_AgentSetPolars._agents["wealth"].to_list()
        )

        # Test with a list of attributes
        result = agents.get(["wealth", "age"])
        assert result[fix1_AgentSetPolars].columns == ["wealth", "age"]
        assert (
            result[fix1_AgentSetPolars]["wealth"].to_list()
            == fix1_AgentSetPolars._agents["wealth"].to_list()
        )
        assert (
            result[fix1_AgentSetPolars]["age"].to_list()
            == fix1_AgentSetPolars._agents["age"].to_list()
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
            fix1_AgentSetPolars._agents["wealth"]
            > fix1_AgentSetPolars._agents["wealth"][0]
        )
        mask1 = (
            fix2_AgentSetPolars._agents["wealth"]
            > fix2_AgentSetPolars._agents["wealth"][0]
        )
        mask_dictionary = {fix1_AgentSetPolars: mask0, fix2_AgentSetPolars: mask1}
        result = agents.get("wealth", mask=mask_dictionary)
        assert (
            result[fix1_AgentSetPolars].to_list()
            == fix1_AgentSetPolars._agents["wealth"].to_list()[1:]
        )
        assert (
            result[fix2_AgentSetPolars].to_list()
            == fix2_AgentSetPolars._agents["wealth"].to_list()[1:]
        )

    def test_remove(
        self,
        fix_AgentsDF: AgentsDF,
        fix3_AgentSetPolars: ExampleAgentSetPolars,
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
            agents._agentsets[0]._agents["unique_id"][0],
            agents._agentsets[1]._agents["unique_id"][0],
        ]
        agentset_polars1 = agents._agentsets[0]
        agentset_polars2 = agents._agentsets[1]
        result = agents.remove(ids, inplace=False)
        assert (
            result._agentsets[0]["unique_id"][0]
            == agentset_polars1._agents.select("unique_id").row(1)[0]
        )
        assert (
            result._agentsets[1].agents["unique_id"][0]
            == agentset_polars2._agents["unique_id"][1]
        )

        # Test if removing an AgentSetDF not present raises ValueError
        with pytest.raises(ValueError):
            result = agents.remove(fix3_AgentSetPolars, inplace=False)

        # Test if removing an ID not present raises KeyError
        assert 0 not in agents._ids
        with pytest.raises(KeyError):
            result = agents.remove(0, inplace=False)

    def test_select(self, fix_AgentsDF: AgentsDF):
        agents = fix_AgentsDF

        # Test with default arguments. Should select all agents
        selected = agents.select(inplace=False)
        active_agents_dict = selected.active_agents
        agents_dict = selected.agents
        assert active_agents_dict.keys() == agents_dict.keys()
        # Using assert to compare all DataFrames in the dictionaries

        assert (
            list(active_agents_dict.values())[0].rows()
            == list(agents_dict.values())[0].rows()
        )

        assert all(
            series.all()
            for series in (
                list(active_agents_dict.values())[1] == list(agents_dict.values())[1]
            )
        )

        # Test with a mask
        mask0 = pl.Series("mask", [True, False, True, True], dtype=pl.Boolean)
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
            # TODO: when pandas support will be dropped, the conversion to python list won't be needed anymore
            return agentset.agents["wealth"] > agentset.agents["wealth"].to_list()[0]

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
        mask0 = pl.Series(
            "mask", [True] + [False] * (len(agents._agentsets[0]) - 1), dtype=pl.Boolean
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
            original_order_0 = agents._agentsets[0].agents["unique_id"].to_list()
            original_order_1 = agents._agentsets[1].agents["unique_id"].to_list()
            agents.shuffle(inplace=True)
            if (
                original_order_0 != agents._agentsets[0].agents["unique_id"].to_list()
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

    def test_step(
        self,
        fix1_AgentSetPolars: ExampleAgentSetPolars,
        fix2_AgentSetPolars: ExampleAgentSetPolars,
        fix_AgentsDF: AgentsDF,
    ):
        previous_wealth_0 = fix1_AgentSetPolars._agents["wealth"].clone()
        previous_wealth_1 = fix2_AgentSetPolars._agents["wealth"].clone()

        agents = fix_AgentsDF
        agents.step()

        assert (
            agents._agentsets[0].agents["wealth"].to_list()
            == (previous_wealth_0 + 1).to_list()
        )
        assert (
            agents._agentsets[1].agents["wealth"].to_list()
            == (previous_wealth_1 + 1).to_list()
        )

    def test__check_ids_presence(
        self,
        fix_AgentsDF: AgentsDF,
        fix1_AgentSetPolars: ExampleAgentSetPolars,
        fix2_AgentSetPolars: ExampleAgentSetPolars,
    ):
        agents = fix_AgentsDF.remove(fix2_AgentSetPolars, inplace=False)
        agents_different_index = deepcopy(fix2_AgentSetPolars)
        result = agents._check_ids_presence([fix1_AgentSetPolars])
        assert result.filter(
            pl.col("unique_id").is_in(fix1_AgentSetPolars._agents["unique_id"])
        )["present"].all()

        assert not result.filter(
            pl.col("unique_id").is_in(agents_different_index._agents["unique_id"])
        )["present"].any()

    def test__check_agentsets_presence(
        self,
        fix_AgentsDF: AgentsDF,
        fix1_AgentSetPolars: ExampleAgentSetPolars,
        fix3_AgentSetPolars: ExampleAgentSetPolars,
    ):
        agents = fix_AgentsDF
        result = agents._check_agentsets_presence(
            [fix1_AgentSetPolars, fix3_AgentSetPolars]
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
            > agents._agentsets[0].agents["wealth"].to_list()[0]
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
                agents._agentsets[0]["unique_id"][0],
                agents._agentsets[1].agents["unique_id"][0],
            ]
        )
        assert result[agents._agentsets[0]].to_list() == [True] + [False] * (
            len(agents._agentsets[0]) - 1
        )
        assert result[agents._agentsets[1]].to_list() == [True] + [False] * (
            len(agents._agentsets[1]) - 1
        )

        # Test with mask = dict[AgentSetDF, AgentMask]
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
        fix1_AgentSetPolars: ExampleAgentSetPolars,
        fix2_AgentSetPolars: ExampleAgentSetPolars,
    ):
        agents = fix_AgentsDF
        result = agents._return_agentsets_list(fix1_AgentSetPolars)
        assert result == [fix1_AgentSetPolars]
        result = agents._return_agentsets_list(
            [fix1_AgentSetPolars, fix2_AgentSetPolars]
        )
        assert result == [fix1_AgentSetPolars, fix2_AgentSetPolars]

    def test___add__(
        self,
        fix1_AgentSetPolars: ExampleAgentSetPolars,
        fix2_AgentSetPolars: ExampleAgentSetPolars,
    ):
        model = ModelDF()
        agents = AgentsDF(model)
        agentset_polars1 = fix1_AgentSetPolars
        agentset_polars2 = fix2_AgentSetPolars

        # Test with a single AgentSetPolars
        result = agents + agentset_polars1
        assert result._agentsets[0] is agentset_polars1
        assert result._ids.to_list() == agentset_polars1._agents["unique_id"].to_list()

        # Test with a single AgentSetPolars same as above
        result = agents + agentset_polars2
        assert result._agentsets[0] is agentset_polars2
        assert result._ids.to_list() == agentset_polars2._agents["unique_id"].to_list()

        # Test with a list of AgentSetDFs
        result = agents + [agentset_polars1, agentset_polars2]
        assert result._agentsets[0] is agentset_polars1
        assert result._agentsets[1] is agentset_polars2
        assert (
            result._ids.to_list()
            == agentset_polars1._agents["unique_id"].to_list()
            + agentset_polars2._agents["unique_id"].to_list()
        )

        # Test if adding the same AgentSetDF raises ValueError
        with pytest.raises(ValueError):
            result + agentset_polars1

    def test___contains__(
        self, fix_AgentsDF: AgentsDF, fix3_AgentSetPolars: ExampleAgentSetPolars
    ):
        # Test with a single value
        agents = fix_AgentsDF
        agentset_polars1 = agents._agentsets[0]

        # Test with an AgentSetDF
        assert agentset_polars1 in agents
        # Test with an AgentSetDF not present
        assert fix3_AgentSetPolars not in agents

        # Test with single id present
        assert agentset_polars1["unique_id"][0] in agents

        # Test with single id not present
        assert 0 not in agents

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
        fix1_AgentSetPolars: ExampleAgentSetPolars,
        fix2_AgentSetPolars: ExampleAgentSetPolars,
    ):
        agents = fix_AgentsDF

        # Test with a single attribute
        assert (
            agents["wealth"][fix1_AgentSetPolars].to_list()
            == fix1_AgentSetPolars._agents["wealth"].to_list()
        )
        assert (
            agents["wealth"][fix2_AgentSetPolars].to_list()
            == fix2_AgentSetPolars._agents["wealth"].to_list()
        )

        # Test with a list of attributes
        result = agents[["wealth", "age"]]
        assert result[fix1_AgentSetPolars].columns == ["wealth", "age"]
        assert (
            result[fix1_AgentSetPolars]["wealth"].to_list()
            == fix1_AgentSetPolars._agents["wealth"].to_list()
        )
        assert (
            result[fix1_AgentSetPolars]["age"].to_list()
            == fix1_AgentSetPolars._agents["age"].to_list()
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
            fix1_AgentSetPolars._agents["wealth"]
            > fix1_AgentSetPolars._agents["wealth"][0]
        )
        mask1 = (
            fix2_AgentSetPolars._agents["wealth"]
            > fix2_AgentSetPolars._agents["wealth"][0]
        )
        mask_dictionary: dict[AgentSetDF, AgentMask] = {
            fix1_AgentSetPolars: mask0,
            fix2_AgentSetPolars: mask1,
        }
        result = agents[mask_dictionary, "wealth"]
        assert (
            result[fix1_AgentSetPolars].to_list()
            == fix1_AgentSetPolars.agents["wealth"].to_list()[1:]
        )
        assert (
            result[fix2_AgentSetPolars].to_list()
            == fix2_AgentSetPolars.agents["wealth"].to_list()[1:]
        )

    def test___iadd__(
        self,
        fix1_AgentSetPolars: ExampleAgentSetPolars,
        fix2_AgentSetPolars: ExampleAgentSetPolars,
    ):
        model = ModelDF()
        agents = AgentsDF(model)
        agentset_polars1 = fix1_AgentSetPolars
        agentset_polars = fix2_AgentSetPolars

        # Test with a single AgentSetPolars
        agents_copy = deepcopy(agents)
        agents_copy += agentset_polars
        assert agents_copy._agentsets[0] is agentset_polars
        assert (
            agents_copy._ids.to_list() == agentset_polars._agents["unique_id"].to_list()
        )

        # Test with a list of AgentSetDFs
        agents_copy = deepcopy(agents)
        agents_copy += [agentset_polars1, agentset_polars]
        assert agents_copy._agentsets[0] is agentset_polars1
        assert agents_copy._agentsets[1] is agentset_polars
        assert (
            agents_copy._ids.to_list()
            == agentset_polars1._agents["unique_id"].to_list()
            + agentset_polars._agents["unique_id"].to_list()
        )

        # Test if adding the same AgentSetDF raises ValueError
        with pytest.raises(ValueError):
            agents_copy += agentset_polars1

    def test___iter__(self, fix_AgentsDF: AgentsDF):
        agents = fix_AgentsDF
        len_agentset0 = len(agents._agentsets[0])
        len_agentset1 = len(agents._agentsets[1])
        for i, agent in enumerate(agents):
            assert isinstance(agent, dict)
            if i < len_agentset0:
                assert agent["unique_id"] == agents._agentsets[0].agents["unique_id"][i]
            else:
                assert (
                    agent["unique_id"]
                    == agents._agentsets[1].agents["unique_id"][i - len_agentset0]
                )
        assert i == len_agentset0 + len_agentset1 - 1

    def test___isub__(
        self,
        fix_AgentsDF: AgentsDF,
        fix1_AgentSetPolars: ExampleAgentSetPolars,
        fix2_AgentSetPolars: ExampleAgentSetPolars,
    ):
        # Test with an AgentSetPolars and a DataFrame
        agents = fix_AgentsDF
        agents -= fix1_AgentSetPolars
        assert agents._agentsets[0] == fix2_AgentSetPolars
        assert len(agents._agentsets) == 1

    def test___len__(
        self,
        fix_AgentsDF: AgentsDF,
        fix1_AgentSetPolars: ExampleAgentSetPolars,
        fix2_AgentSetPolars: ExampleAgentSetPolars,
    ):
        assert len(fix_AgentsDF) == len(fix1_AgentSetPolars) + len(fix2_AgentSetPolars)

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
        mask0 = pl.Series(
            "mask", [True] + [False] * (len(agents._agentsets[0]) - 1), dtype=pl.Boolean
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
        fix1_AgentSetPolars: ExampleAgentSetPolars,
        fix2_AgentSetPolars: ExampleAgentSetPolars,
    ):
        # Test with an AgentSetPolars and a DataFrame
        result = fix_AgentsDF - fix1_AgentSetPolars
        assert isinstance(result._agentsets[0], ExampleAgentSetPolars)
        assert len(result._agentsets) == 1

    def test_agents(
        self,
        fix_AgentsDF: AgentsDF,
        fix1_AgentSetPolars: ExampleAgentSetPolars,
        fix2_AgentSetPolars: ExampleAgentSetPolars,
    ):
        assert isinstance(fix_AgentsDF.agents, dict)
        assert len(fix_AgentsDF.agents) == 2
        assert fix_AgentsDF.agents[fix1_AgentSetPolars] is fix1_AgentSetPolars._agents
        assert fix_AgentsDF.agents[fix2_AgentSetPolars] is fix2_AgentSetPolars._agents

        # Test agents.setter
        fix_AgentsDF.agents = [fix1_AgentSetPolars, fix2_AgentSetPolars]
        assert fix_AgentsDF._agentsets[0] == fix1_AgentSetPolars
        assert fix_AgentsDF._agentsets[1] == fix2_AgentSetPolars

    def test_active_agents(self, fix_AgentsDF: AgentsDF):
        agents = fix_AgentsDF

        # Test with select
        mask0 = (
            agents._agentsets[0].agents["wealth"]
            > agents._agentsets[0].agents["wealth"].to_list()[0]
        )
        mask1 = (
            agents._agentsets[1].agents["wealth"]
            > agents._agentsets[1].agents["wealth"].to_list()[0]
        )
        mask_dictionary = {agents._agentsets[0]: mask0, agents._agentsets[1]: mask1}

        agents1 = agents.select(mask=mask_dictionary, inplace=False)

        result = agents1.active_agents
        assert isinstance(result, dict)
        assert isinstance(result[agents1._agentsets[0]], pl.DataFrame)
        assert isinstance(result[agents1._agentsets[1]], pl.DataFrame)

        assert all(
            series.all()
            for series in (
                result[agents1._agentsets[0]]
                == agents1._agentsets[0]._agents.filter(mask0)
            )
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
        assert isinstance(result[agents1._agentsets[0]], pl.DataFrame)
        assert isinstance(result[agents1._agentsets[1]], pl.DataFrame)
        assert all(
            series.all()
            for series in (
                result[agents1._agentsets[0]]
                == agents1._agentsets[0]._agents.filter(mask0)
            )
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
        assert isinstance(result[ExampleAgentSetPolars], AgentsDF)

        assert (
            result[ExampleAgentSetPolars]._agentsets[0].agents.rows()
            == agents._agentsets[1].agents.rows()
        )

    def test_inactive_agents(self, fix_AgentsDF: AgentsDF):
        agents = fix_AgentsDF

        # Test with select
        mask0 = (
            agents._agentsets[0].agents["wealth"]
            > agents._agentsets[0].agents["wealth"].to_list()[0]
        )
        mask1 = (
            agents._agentsets[1].agents["wealth"]
            > agents._agentsets[1].agents["wealth"][0]
        )
        mask_dictionary = {agents._agentsets[0]: mask0, agents._agentsets[1]: mask1}
        agents1 = agents.select(mask=mask_dictionary, inplace=False)
        result = agents1.inactive_agents
        assert isinstance(result, dict)
        assert isinstance(result[agents1._agentsets[0]], pl.DataFrame)
        assert isinstance(result[agents1._agentsets[1]], pl.DataFrame)
        assert all(
            series.all()
            for series in (
                result[agents1._agentsets[0]]
                == agents1._agentsets[0].select(mask0, negate=True).active_agents
            )
        )
        assert all(
            series.all()
            for series in (
                result[agents1._agentsets[1]]
                == agents1._agentsets[1].select(mask1, negate=True).active_agents
            )
        )
