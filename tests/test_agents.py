from copy import copy, deepcopy

import polars as pl
import pytest

from mesa_frames import AgentSetRegistry, Model
from mesa_frames import AgentSet
from mesa_frames.types_ import AgentMask
from tests.test_agentset import (
    ExampleAgentSet,
    ExampleAgentSetNoWealth,
    fix1_AgentSet_no_wealth,
    fix1_AgentSet,
    fix2_AgentSet,
    fix3_AgentSet,
)


@pytest.fixture
def fix_AgentSetRegistry(
    fix1_AgentSet: ExampleAgentSet,
    fix2_AgentSet: ExampleAgentSet,
) -> AgentSetRegistry:
    model = Model()
    agents = AgentSetRegistry(model)
    agents.add([fix1_AgentSet, fix2_AgentSet])
    return agents


class Test_AgentSetRegistry:
    def test___init__(self):
        model = Model()
        agents = AgentSetRegistry(model)
        assert agents.model == model
        assert isinstance(agents._agentsets, list)
        assert len(agents._agentsets) == 0
        assert isinstance(agents._ids, pl.Series)
        assert agents._ids.is_empty()
        assert agents._ids.name == "unique_id"

    def test_add(
        self,
        fix1_AgentSet: ExampleAgentSet,
        fix2_AgentSet: ExampleAgentSet,
    ):
        model = Model()
        agents = AgentSetRegistry(model)
        agentset_polars1 = fix1_AgentSet
        agentset_polars2 = fix2_AgentSet

        # Test with a single AgentSet
        result = agents.add(agentset_polars1, inplace=False)
        assert result._agentsets[0] is agentset_polars1
        assert result._ids.to_list() == agentset_polars1._df["unique_id"].to_list()

        # Test with a list of AgentSets
        result = agents.add([agentset_polars1, agentset_polars2], inplace=True)
        assert result._agentsets[0] is agentset_polars1
        assert result._agentsets[1] is agentset_polars2
        assert (
            result._ids.to_list()
            == agentset_polars1._df["unique_id"].to_list()
            + agentset_polars2._df["unique_id"].to_list()
        )

        # Test if adding the same AgentSet raises ValueError
        with pytest.raises(ValueError):
            agents.add(agentset_polars1, inplace=False)

    def test_contains(
        self,
        fix1_AgentSet: ExampleAgentSet,
        fix2_AgentSet: ExampleAgentSet,
        fix3_AgentSet: ExampleAgentSet,
        fix_AgentSetRegistry: AgentSetRegistry,
    ):
        agents = fix_AgentSetRegistry
        agentset_polars1 = agents._agentsets[0]

        # Test with an AgentSet
        assert agents.contains(agentset_polars1)
        assert agents.contains(fix1_AgentSet)
        assert agents.contains(fix2_AgentSet)

        # Test with an AgentSet not present
        assert not agents.contains(fix3_AgentSet)

        # Test with an iterable of AgentSets
        assert agents.contains([agentset_polars1, fix3_AgentSet]).to_list() == [
            True,
            False,
        ]

        # Test with empty iterable - returns True
        assert agents.contains([])

        # Test with single id
        assert agents.contains(agentset_polars1["unique_id"][0])

        # Test with a list of ids
        assert agents.contains([agentset_polars1["unique_id"][0], 0]).to_list() == [
            True,
            False,
        ]

    def test_copy(self, fix_AgentSetRegistry: AgentSetRegistry):
        agents = fix_AgentSetRegistry
        agents.test_list = [[1, 2, 3]]

        # Test with deep=False
        agents2 = agents.copy(deep=False)
        agents2.test_list[0].append(4)
        assert agents.test_list[0][-1] == agents2.test_list[0][-1]
        assert agents.model == agents2.model
        assert agents._agentsets[0] == agents2._agentsets[0]
        assert (agents._ids == agents2._ids).all()

        # Test with deep=True
        agents2 = fix_AgentSetRegistry.copy(deep=True)
        agents2.test_list[0].append(4)
        assert agents.test_list[-1] != agents2.test_list[-1]
        assert agents.model == agents2.model
        assert agents._agentsets[0] != agents2._agentsets[0]
        assert (agents._ids == agents2._ids).all()

    def test_discard(
        self, fix_AgentSetRegistry: AgentSetRegistry, fix2_AgentSet: ExampleAgentSet
    ):
        agents = fix_AgentSetRegistry
        # Test with a single AgentSet
        agentset_polars2 = agents._agentsets[1]
        result = agents.discard(agents._agentsets[0], inplace=False)
        assert isinstance(result._agentsets[0], ExampleAgentSet)
        assert len(result._agentsets) == 1

        # Test with a list of AgentSets
        result = agents.discard(agents._agentsets.copy(), inplace=False)
        assert len(result._agentsets) == 0

        # Test with IDs
        ids = [
            agents._agentsets[0]._df["unique_id"][0],
            agents._agentsets[1]._df["unique_id"][0],
        ]
        agentset_polars1 = agents._agentsets[0]
        agentset_polars2 = agents._agentsets[1]
        result = agents.discard(ids, inplace=False)
        assert (
            result._agentsets[0]["unique_id"][0]
            == agentset_polars1._df.select("unique_id").row(1)[0]
        )
        assert (
            result._agentsets[1].df["unique_id"][0]
            == agentset_polars2._df["unique_id"][1]
        )

        # Test if removing an AgentSet not present raises ValueError
        result = agents.discard(fix2_AgentSet, inplace=False)

        # Test if removing an ID not present raises KeyError
        assert 0 not in agents._ids
        result = agents.discard(0, inplace=False)

    def test_do(self, fix_AgentSetRegistry: AgentSetRegistry):
        agents = fix_AgentSetRegistry

        expected_result_0 = agents._agentsets[0].df["wealth"]
        expected_result_0 += 1

        expected_result_1 = agents._agentsets[1].df["wealth"]
        expected_result_1 += 1

        # Test with no return_results, no mask, inplace
        agents.do("add_wealth", 1)
        assert (
            agents._agentsets[0].df["wealth"].to_list() == expected_result_0.to_list()
        )
        assert (
            agents._agentsets[1].df["wealth"].to_list() == expected_result_1.to_list()
        )

        # Test with return_results=True, no mask, inplace
        expected_result_0 = agents._agentsets[0].df["wealth"]
        expected_result_0 += 1

        expected_result_1 = agents._agentsets[1].df["wealth"]
        expected_result_1 += 1
        assert agents.do("add_wealth", 1, return_results=True) == {
            agents._agentsets[0]: None,
            agents._agentsets[1]: None,
        }
        assert (
            agents._agentsets[0].df["wealth"].to_list() == expected_result_0.to_list()
        )
        assert (
            agents._agentsets[1].df["wealth"].to_list() == expected_result_1.to_list()
        )

        # Test with a mask, inplace
        mask0 = agents._agentsets[0].df["wealth"] > 10  # No agent should be selected
        mask1 = agents._agentsets[1].df["wealth"] > 10  # All agents should be selected
        mask_dictionary = {agents._agentsets[0]: mask0, agents._agentsets[1]: mask1}

        expected_result_0 = agents._agentsets[0].df["wealth"]
        expected_result_1 = agents._agentsets[1].df["wealth"]
        expected_result_1 += 1

        agents.do("add_wealth", 1, mask=mask_dictionary)
        assert (
            agents._agentsets[0].df["wealth"].to_list() == expected_result_0.to_list()
        )
        assert (
            agents._agentsets[1].df["wealth"].to_list() == expected_result_1.to_list()
        )

    def test_get(
        self,
        fix_AgentSetRegistry: AgentSetRegistry,
        fix1_AgentSet: ExampleAgentSet,
        fix2_AgentSet: ExampleAgentSet,
        fix1_AgentSet_no_wealth: ExampleAgentSetNoWealth,
    ):
        agents = fix_AgentSetRegistry

        # Test with a single attribute
        assert (
            agents.get("wealth")[fix1_AgentSet].to_list()
            == fix1_AgentSet._df["wealth"].to_list()
        )
        assert (
            agents.get("wealth")[fix2_AgentSet].to_list()
            == fix2_AgentSet._df["wealth"].to_list()
        )

        # Test with a list of attributes
        result = agents.get(["wealth", "age"])
        assert result[fix1_AgentSet].columns == ["wealth", "age"]
        assert (
            result[fix1_AgentSet]["wealth"].to_list()
            == fix1_AgentSet._df["wealth"].to_list()
        )
        assert (
            result[fix1_AgentSet]["age"].to_list() == fix1_AgentSet._df["age"].to_list()
        )

        assert result[fix2_AgentSet].columns == ["wealth", "age"]
        assert (
            result[fix2_AgentSet]["wealth"].to_list()
            == fix2_AgentSet._df["wealth"].to_list()
        )
        assert (
            result[fix2_AgentSet]["age"].to_list() == fix2_AgentSet._df["age"].to_list()
        )

        # Test with a single attribute and a mask
        mask0 = fix1_AgentSet._df["wealth"] > fix1_AgentSet._df["wealth"][0]
        mask1 = fix2_AgentSet._df["wealth"] > fix2_AgentSet._df["wealth"][0]
        mask_dictionary = {fix1_AgentSet: mask0, fix2_AgentSet: mask1}
        result = agents.get("wealth", mask=mask_dictionary)
        assert (
            result[fix1_AgentSet].to_list() == fix1_AgentSet._df["wealth"].to_list()[1:]
        )
        assert (
            result[fix2_AgentSet].to_list() == fix2_AgentSet._df["wealth"].to_list()[1:]
        )

        # Test heterogeneous agent sets (different columns)
        # This tests the fix for the bug where agents_df["column"] would raise
        # ColumnNotFoundError when some agent sets didn't have that column.

        # Create a new AgentSetRegistry with heterogeneous agent sets
        model = Model()
        hetero_agents = AgentSetRegistry(model)
        hetero_agents.add([fix1_AgentSet, fix1_AgentSet_no_wealth])

        # Test 1: Access column that exists in only one agent set
        result_wealth = hetero_agents.get("wealth")
        assert len(result_wealth) == 1, (
            "Should only return agent sets that have 'wealth'"
        )
        assert fix1_AgentSet in result_wealth, (
            "Should include the agent set with wealth"
        )
        assert fix1_AgentSet_no_wealth not in result_wealth, (
            "Should not include agent set without wealth"
        )
        assert result_wealth[fix1_AgentSet].to_list() == [1, 2, 3, 4]

        # Test 2: Access column that exists in all agent sets
        result_age = hetero_agents.get("age")
        assert len(result_age) == 2, "Should return both agent sets that have 'age'"
        assert fix1_AgentSet in result_age
        assert fix1_AgentSet_no_wealth in result_age
        assert result_age[fix1_AgentSet].to_list() == [10, 20, 30, 40]
        assert result_age[fix1_AgentSet_no_wealth].to_list() == [1, 2, 3, 4]

        # Test 3: Access column that exists in no agent sets
        result_nonexistent = hetero_agents.get("nonexistent_column")
        assert len(result_nonexistent) == 0, (
            "Should return empty dict for non-existent column"
        )

        # Test 4: Access multiple columns (mixed availability)
        result_multi = hetero_agents.get(["wealth", "age"])
        assert len(result_multi) == 1, (
            "Should only include agent sets that have ALL requested columns"
        )
        assert fix1_AgentSet in result_multi
        assert fix1_AgentSet_no_wealth not in result_multi
        assert result_multi[fix1_AgentSet].columns == ["wealth", "age"]

        # Test 5: Access multiple columns where some exist in different sets
        result_mixed = hetero_agents.get(["age", "income"])
        assert len(result_mixed) == 1, (
            "Should only include agent set that has both 'age' and 'income'"
        )
        assert fix1_AgentSet_no_wealth in result_mixed
        assert fix1_AgentSet not in result_mixed

        # Test 6: Test via __getitem__ syntax (the original bug report case)
        wealth_via_getitem = hetero_agents["wealth"]
        assert len(wealth_via_getitem) == 1
        assert fix1_AgentSet in wealth_via_getitem
        assert wealth_via_getitem[fix1_AgentSet].to_list() == [1, 2, 3, 4]

        # Test 7: Test get(None) - should return all columns for all agent sets
        result_none = hetero_agents.get(None)
        assert len(result_none) == 2, (
            "Should return both agent sets when attr_names=None"
        )
        assert fix1_AgentSet in result_none
        assert fix1_AgentSet_no_wealth in result_none

        # Verify each agent set returns all its columns (excluding unique_id)
        wealth_set_result = result_none[fix1_AgentSet]
        assert isinstance(wealth_set_result, pl.DataFrame), (
            "Should return DataFrame when attr_names=None"
        )
        expected_wealth_cols = {"wealth", "age"}  # unique_id should be excluded
        assert set(wealth_set_result.columns) == expected_wealth_cols

        no_wealth_set_result = result_none[fix1_AgentSet_no_wealth]
        assert isinstance(no_wealth_set_result, pl.DataFrame), (
            "Should return DataFrame when attr_names=None"
        )
        expected_no_wealth_cols = {"income", "age"}  # unique_id should be excluded
        assert set(no_wealth_set_result.columns) == expected_no_wealth_cols

    def test_remove(
        self,
        fix_AgentSetRegistry: AgentSetRegistry,
        fix3_AgentSet: ExampleAgentSet,
    ):
        agents = fix_AgentSetRegistry

        # Test with a single AgentSet
        agentset_polars = agents._agentsets[1]
        result = agents.remove(agents._agentsets[0], inplace=False)
        assert isinstance(result._agentsets[0], ExampleAgentSet)
        assert len(result._agentsets) == 1

        # Test with a list of AgentSets
        result = agents.remove(agents._agentsets.copy(), inplace=False)
        assert len(result._agentsets) == 0

        # Test with IDs
        ids = [
            agents._agentsets[0]._df["unique_id"][0],
            agents._agentsets[1]._df["unique_id"][0],
        ]
        agentset_polars1 = agents._agentsets[0]
        agentset_polars2 = agents._agentsets[1]
        result = agents.remove(ids, inplace=False)
        assert (
            result._agentsets[0]["unique_id"][0]
            == agentset_polars1._df.select("unique_id").row(1)[0]
        )
        assert (
            result._agentsets[1].df["unique_id"][0]
            == agentset_polars2._df["unique_id"][1]
        )

        # Test if removing an AgentSet not present raises ValueError
        with pytest.raises(ValueError):
            result = agents.remove(fix3_AgentSet, inplace=False)

        # Test if removing an ID not present raises KeyError
        assert 0 not in agents._ids
        with pytest.raises(KeyError):
            result = agents.remove(0, inplace=False)

    def test_select(self, fix_AgentSetRegistry: AgentSetRegistry):
        agents = fix_AgentSetRegistry

        # Test with default arguments. Should select all agents
        selected = agents.select(inplace=False)
        active_agents_dict = selected.active_agents
        agents_dict = selected.df
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

        def filter_func(agentset: AgentSet) -> pl.Series:
            return agentset.df["wealth"] > agentset.df["wealth"].to_list()[0]

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

    def test_set(self, fix_AgentSetRegistry: AgentSetRegistry):
        agents = fix_AgentSetRegistry

        # Test with a single attribute
        result = agents.set("wealth", 0, inplace=False)
        assert result._agentsets[0].df["wealth"].to_list() == [0] * len(
            agents._agentsets[0]
        )
        assert result._agentsets[1].df["wealth"].to_list() == [0] * len(
            agents._agentsets[1]
        )

        # Test with a list of attributes
        agents.set(["wealth", "age"], 1, inplace=True)
        assert agents._agentsets[0].df["wealth"].to_list() == [1] * len(
            agents._agentsets[0]
        )
        assert agents._agentsets[0].df["age"].to_list() == [1] * len(
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
        assert result._agentsets[0].df["wealth"].to_list() == [0] + [1] * (
            len(agents._agentsets[0]) - 1
        )
        assert result._agentsets[1].df["wealth"].to_list() == [0] + [1] * (
            len(agents._agentsets[1]) - 1
        )

        # Test with a dictionary
        agents.set(
            {agents._agentsets[0]: {"wealth": 0}, agents._agentsets[1]: {"wealth": 1}},
            inplace=True,
        )
        assert agents._agentsets[0].df["wealth"].to_list() == [0] * len(
            agents._agentsets[0]
        )
        assert agents._agentsets[1].df["wealth"].to_list() == [1] * len(
            agents._agentsets[1]
        )

    def test_shuffle(self, fix_AgentSetRegistry: AgentSetRegistry):
        agents = fix_AgentSetRegistry
        for _ in range(100):
            original_order_0 = agents._agentsets[0].df["unique_id"].to_list()
            original_order_1 = agents._agentsets[1].df["unique_id"].to_list()
            agents.shuffle(inplace=True)
            if (
                original_order_0 != agents._agentsets[0].df["unique_id"].to_list()
                and original_order_1 != agents._agentsets[1].df["unique_id"].to_list()
            ):
                return
        assert False

    def test_sort(self, fix_AgentSetRegistry: AgentSetRegistry):
        agents = fix_AgentSetRegistry
        agents.sort("wealth", ascending=False, inplace=True)
        assert pl.Series(agents._agentsets[0].df["wealth"]).is_sorted(descending=True)
        assert pl.Series(agents._agentsets[1].df["wealth"]).is_sorted(descending=True)

    def test_step(
        self,
        fix1_AgentSet: ExampleAgentSet,
        fix2_AgentSet: ExampleAgentSet,
        fix_AgentSetRegistry: AgentSetRegistry,
    ):
        previous_wealth_0 = fix1_AgentSet._df["wealth"].clone()
        previous_wealth_1 = fix2_AgentSet._df["wealth"].clone()

        agents = fix_AgentSetRegistry
        agents.step()

        assert (
            agents._agentsets[0].df["wealth"].to_list()
            == (previous_wealth_0 + 1).to_list()
        )
        assert (
            agents._agentsets[1].df["wealth"].to_list()
            == (previous_wealth_1 + 1).to_list()
        )

    def test__check_ids_presence(
        self,
        fix_AgentSetRegistry: AgentSetRegistry,
        fix1_AgentSet: ExampleAgentSet,
        fix2_AgentSet: ExampleAgentSet,
    ):
        agents = fix_AgentSetRegistry.remove(fix2_AgentSet, inplace=False)
        agents_different_index = deepcopy(fix2_AgentSet)
        result = agents._check_ids_presence([fix1_AgentSet])
        assert result.filter(pl.col("unique_id").is_in(fix1_AgentSet._df["unique_id"]))[
            "present"
        ].all()

        assert not result.filter(
            pl.col("unique_id").is_in(agents_different_index._df["unique_id"])
        )["present"].any()

    def test__check_agentsets_presence(
        self,
        fix_AgentSetRegistry: AgentSetRegistry,
        fix1_AgentSet: ExampleAgentSet,
        fix3_AgentSet: ExampleAgentSet,
    ):
        agents = fix_AgentSetRegistry
        result = agents._check_agentsets_presence([fix1_AgentSet, fix3_AgentSet])
        assert result[0]
        assert not result[1]

    def test__get_bool_masks(self, fix_AgentSetRegistry: AgentSetRegistry):
        agents = fix_AgentSetRegistry
        # Test with mask = None
        result = agents._get_bool_masks(mask=None)
        truth_value = True
        for i, mask in enumerate(result.values()):
            if isinstance(mask, pl.Expr):
                mask = agents._agentsets[i]._df.select(mask).to_series()
            truth_value &= mask.all()
        assert truth_value

        # Test with mask = "all"
        result = agents._get_bool_masks(mask="all")
        truth_value = True
        for i, mask in enumerate(result.values()):
            if isinstance(mask, pl.Expr):
                mask = agents._agentsets[i]._df.select(mask).to_series()
            truth_value &= mask.all()
        assert truth_value

        # Test with mask = "active"
        mask0 = (
            agents._agentsets[0].df["wealth"]
            > agents._agentsets[0].df["wealth"].to_list()[0]
        )
        mask1 = agents._agentsets[1].df["wealth"] > agents._agentsets[1].df["wealth"][0]
        mask_dictionary = {agents._agentsets[0]: mask0, agents._agentsets[1]: mask1}
        agents.select(mask=mask_dictionary)
        result = agents._get_bool_masks(mask="active")
        assert result[agents._agentsets[0]].to_list() == mask0.to_list()
        assert result[agents._agentsets[1]].to_list() == mask1.to_list()

        # Test with mask = IdsLike
        result = agents._get_bool_masks(
            mask=[
                agents._agentsets[0]["unique_id"][0],
                agents._agentsets[1].df["unique_id"][0],
            ]
        )
        assert result[agents._agentsets[0]].to_list() == [True] + [False] * (
            len(agents._agentsets[0]) - 1
        )
        assert result[agents._agentsets[1]].to_list() == [True] + [False] * (
            len(agents._agentsets[1]) - 1
        )

        # Test with mask = dict[AgentSet, AgentMask]
        result = agents._get_bool_masks(mask=mask_dictionary)
        assert result[agents._agentsets[0]].to_list() == mask0.to_list()
        assert result[agents._agentsets[1]].to_list() == mask1.to_list()

    def test__get_obj(self, fix_AgentSetRegistry: AgentSetRegistry):
        agents = fix_AgentSetRegistry
        assert agents._get_obj(inplace=True) is agents
        assert agents._get_obj(inplace=False) is not agents

    def test__return_agentsets_list(
        self,
        fix_AgentSetRegistry: AgentSetRegistry,
        fix1_AgentSet: ExampleAgentSet,
        fix2_AgentSet: ExampleAgentSet,
    ):
        agents = fix_AgentSetRegistry
        result = agents._return_agentsets_list(fix1_AgentSet)
        assert result == [fix1_AgentSet]
        result = agents._return_agentsets_list([fix1_AgentSet, fix2_AgentSet])
        assert result == [fix1_AgentSet, fix2_AgentSet]

    def test___add__(
        self,
        fix1_AgentSet: ExampleAgentSet,
        fix2_AgentSet: ExampleAgentSet,
    ):
        model = Model()
        agents = AgentSetRegistry(model)
        agentset_polars1 = fix1_AgentSet
        agentset_polars2 = fix2_AgentSet

        # Test with a single AgentSet
        result = agents + agentset_polars1
        assert result._agentsets[0] is agentset_polars1
        assert result._ids.to_list() == agentset_polars1._df["unique_id"].to_list()

        # Test with a single AgentSet same as above
        result = agents + agentset_polars2
        assert result._agentsets[0] is agentset_polars2
        assert result._ids.to_list() == agentset_polars2._df["unique_id"].to_list()

        # Test with a list of AgentSets
        result = agents + [agentset_polars1, agentset_polars2]
        assert result._agentsets[0] is agentset_polars1
        assert result._agentsets[1] is agentset_polars2
        assert (
            result._ids.to_list()
            == agentset_polars1._df["unique_id"].to_list()
            + agentset_polars2._df["unique_id"].to_list()
        )

        # Test if adding the same AgentSet raises ValueError
        with pytest.raises(ValueError):
            result + agentset_polars1

    def test___contains__(
        self, fix_AgentSetRegistry: AgentSetRegistry, fix3_AgentSet: ExampleAgentSet
    ):
        # Test with a single value
        agents = fix_AgentSetRegistry
        agentset_polars1 = agents._agentsets[0]

        # Test with an AgentSet
        assert agentset_polars1 in agents
        # Test with an AgentSet not present
        assert fix3_AgentSet not in agents

        # Test with single id present
        assert agentset_polars1["unique_id"][0] in agents

        # Test with single id not present
        assert 0 not in agents

    def test___copy__(self, fix_AgentSetRegistry: AgentSetRegistry):
        agents = fix_AgentSetRegistry
        agents.test_list = [[1, 2, 3]]

        # Test with deep=False
        agents2 = copy(agents)
        agents2.test_list[0].append(4)
        assert agents.test_list[0][-1] == agents2.test_list[0][-1]
        assert agents.model == agents2.model
        assert agents._agentsets[0] == agents2._agentsets[0]
        assert (agents._ids == agents2._ids).all()

    def test___deepcopy__(self, fix_AgentSetRegistry: AgentSetRegistry):
        agents = fix_AgentSetRegistry
        agents.test_list = [[1, 2, 3]]

        agents2 = deepcopy(agents)
        agents2.test_list[0].append(4)
        assert agents.test_list[-1] != agents2.test_list[-1]
        assert agents.model == agents2.model
        assert agents._agentsets[0] != agents2._agentsets[0]
        assert (agents._ids == agents2._ids).all()

    def test___getattr__(self, fix_AgentSetRegistry: AgentSetRegistry):
        agents = fix_AgentSetRegistry
        assert isinstance(agents.model, Model)
        result = agents.wealth
        assert (
            result[agents._agentsets[0]].to_list()
            == agents._agentsets[0].df["wealth"].to_list()
        )
        assert (
            result[agents._agentsets[1]].to_list()
            == agents._agentsets[1].df["wealth"].to_list()
        )

    def test___getitem__(
        self,
        fix_AgentSetRegistry: AgentSetRegistry,
        fix1_AgentSet: ExampleAgentSet,
        fix2_AgentSet: ExampleAgentSet,
    ):
        agents = fix_AgentSetRegistry

        # Test with a single attribute
        assert (
            agents["wealth"][fix1_AgentSet].to_list()
            == fix1_AgentSet._df["wealth"].to_list()
        )
        assert (
            agents["wealth"][fix2_AgentSet].to_list()
            == fix2_AgentSet._df["wealth"].to_list()
        )

        # Test with a list of attributes
        result = agents[["wealth", "age"]]
        assert result[fix1_AgentSet].columns == ["wealth", "age"]
        assert (
            result[fix1_AgentSet]["wealth"].to_list()
            == fix1_AgentSet._df["wealth"].to_list()
        )
        assert (
            result[fix1_AgentSet]["age"].to_list() == fix1_AgentSet._df["age"].to_list()
        )
        assert result[fix2_AgentSet].columns == ["wealth", "age"]
        assert (
            result[fix2_AgentSet]["wealth"].to_list()
            == fix2_AgentSet._df["wealth"].to_list()
        )
        assert (
            result[fix2_AgentSet]["age"].to_list() == fix2_AgentSet._df["age"].to_list()
        )

        # Test with a single attribute and a mask
        mask0 = fix1_AgentSet._df["wealth"] > fix1_AgentSet._df["wealth"][0]
        mask1 = fix2_AgentSet._df["wealth"] > fix2_AgentSet._df["wealth"][0]
        mask_dictionary: dict[AgentSet, AgentMask] = {
            fix1_AgentSet: mask0,
            fix2_AgentSet: mask1,
        }
        result = agents[mask_dictionary, "wealth"]
        assert (
            result[fix1_AgentSet].to_list() == fix1_AgentSet.df["wealth"].to_list()[1:]
        )
        assert (
            result[fix2_AgentSet].to_list() == fix2_AgentSet.df["wealth"].to_list()[1:]
        )

    def test___iadd__(
        self,
        fix1_AgentSet: ExampleAgentSet,
        fix2_AgentSet: ExampleAgentSet,
    ):
        model = Model()
        agents = AgentSetRegistry(model)
        agentset_polars1 = fix1_AgentSet
        agentset_polars = fix2_AgentSet

        # Test with a single AgentSet
        agents_copy = deepcopy(agents)
        agents_copy += agentset_polars
        assert agents_copy._agentsets[0] is agentset_polars
        assert agents_copy._ids.to_list() == agentset_polars._df["unique_id"].to_list()

        # Test with a list of AgentSets
        agents_copy = deepcopy(agents)
        agents_copy += [agentset_polars1, agentset_polars]
        assert agents_copy._agentsets[0] is agentset_polars1
        assert agents_copy._agentsets[1] is agentset_polars
        assert (
            agents_copy._ids.to_list()
            == agentset_polars1._df["unique_id"].to_list()
            + agentset_polars._df["unique_id"].to_list()
        )

        # Test if adding the same AgentSet raises ValueError
        with pytest.raises(ValueError):
            agents_copy += agentset_polars1

    def test___iter__(self, fix_AgentSetRegistry: AgentSetRegistry):
        agents = fix_AgentSetRegistry
        len_agentset0 = len(agents._agentsets[0])
        len_agentset1 = len(agents._agentsets[1])
        for i, agent in enumerate(agents):
            assert isinstance(agent, dict)
            if i < len_agentset0:
                assert agent["unique_id"] == agents._agentsets[0].df["unique_id"][i]
            else:
                assert (
                    agent["unique_id"]
                    == agents._agentsets[1].df["unique_id"][i - len_agentset0]
                )
        assert i == len_agentset0 + len_agentset1 - 1

    def test___isub__(
        self,
        fix_AgentSetRegistry: AgentSetRegistry,
        fix1_AgentSet: ExampleAgentSet,
        fix2_AgentSet: ExampleAgentSet,
    ):
        # Test with an AgentSet and a DataFrame
        agents = fix_AgentSetRegistry
        agents -= fix1_AgentSet
        assert agents._agentsets[0] == fix2_AgentSet
        assert len(agents._agentsets) == 1

    def test___len__(
        self,
        fix_AgentSetRegistry: AgentSetRegistry,
        fix1_AgentSet: ExampleAgentSet,
        fix2_AgentSet: ExampleAgentSet,
    ):
        assert len(fix_AgentSetRegistry) == len(fix1_AgentSet) + len(fix2_AgentSet)

    def test___repr__(self, fix_AgentSetRegistry: AgentSetRegistry):
        repr(fix_AgentSetRegistry)

    def test___reversed__(self, fix2_AgentSet: AgentSetRegistry):
        agents = fix2_AgentSet
        reversed_wealth = []
        for agent in reversed(list(agents)):
            reversed_wealth.append(agent["wealth"])
        assert reversed_wealth == list(reversed(agents["wealth"]))

    def test___setitem__(self, fix_AgentSetRegistry: AgentSetRegistry):
        agents = fix_AgentSetRegistry

        # Test with a single attribute
        agents["wealth"] = 0
        assert agents._agentsets[0].df["wealth"].to_list() == [0] * len(
            agents._agentsets[0]
        )
        assert agents._agentsets[1].df["wealth"].to_list() == [0] * len(
            agents._agentsets[1]
        )

        # Test with a list of attributes
        agents[["wealth", "age"]] = 1
        assert agents._agentsets[0].df["wealth"].to_list() == [1] * len(
            agents._agentsets[0]
        )
        assert agents._agentsets[0].df["age"].to_list() == [1] * len(
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
        assert agents._agentsets[0].df["wealth"].to_list() == [0] + [1] * (
            len(agents._agentsets[0]) - 1
        )
        assert agents._agentsets[1].df["wealth"].to_list() == [0] + [1] * (
            len(agents._agentsets[1]) - 1
        )

    def test___str__(self, fix_AgentSetRegistry: AgentSetRegistry):
        str(fix_AgentSetRegistry)

    def test___sub__(
        self,
        fix_AgentSetRegistry: AgentSetRegistry,
        fix1_AgentSet: ExampleAgentSet,
        fix2_AgentSet: ExampleAgentSet,
    ):
        # Test with an AgentSet and a DataFrame
        result = fix_AgentSetRegistry - fix1_AgentSet
        assert isinstance(result._agentsets[0], ExampleAgentSet)
        assert len(result._agentsets) == 1

    def test_agents(
        self,
        fix_AgentSetRegistry: AgentSetRegistry,
        fix1_AgentSet: ExampleAgentSet,
        fix2_AgentSet: ExampleAgentSet,
    ):
        assert isinstance(fix_AgentSetRegistry.df, dict)
        assert len(fix_AgentSetRegistry.df) == 2
        assert fix_AgentSetRegistry.df[fix1_AgentSet] is fix1_AgentSet._df
        assert fix_AgentSetRegistry.df[fix2_AgentSet] is fix2_AgentSet._df

        # Test agents.setter
        fix_AgentSetRegistry.df = [fix1_AgentSet, fix2_AgentSet]
        assert fix_AgentSetRegistry._agentsets[0] == fix1_AgentSet
        assert fix_AgentSetRegistry._agentsets[1] == fix2_AgentSet

    def test_active_agents(self, fix_AgentSetRegistry: AgentSetRegistry):
        agents = fix_AgentSetRegistry

        # Test with select
        mask0 = (
            agents._agentsets[0].df["wealth"]
            > agents._agentsets[0].df["wealth"].to_list()[0]
        )
        mask1 = (
            agents._agentsets[1].df["wealth"]
            > agents._agentsets[1].df["wealth"].to_list()[0]
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
                result[agents1._agentsets[0]] == agents1._agentsets[0]._df.filter(mask0)
            )
        )

        assert all(
            series.all()
            for series in (
                result[agents1._agentsets[1]] == agents1._agentsets[1]._df.filter(mask1)
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
                result[agents1._agentsets[0]] == agents1._agentsets[0]._df.filter(mask0)
            )
        )
        assert all(
            series.all()
            for series in (
                result[agents1._agentsets[1]] == agents1._agentsets[1]._df.filter(mask1)
            )
        )

    def test_agentsets_by_type(self, fix_AgentSetRegistry: AgentSetRegistry):
        agents = fix_AgentSetRegistry

        result = agents.agentsets_by_type
        assert isinstance(result, dict)
        assert isinstance(result[ExampleAgentSet], AgentSetRegistry)

        assert (
            result[ExampleAgentSet]._agentsets[0].df.rows()
            == agents._agentsets[1].df.rows()
        )

    def test_inactive_agents(self, fix_AgentSetRegistry: AgentSetRegistry):
        agents = fix_AgentSetRegistry

        # Test with select
        mask0 = (
            agents._agentsets[0].df["wealth"]
            > agents._agentsets[0].df["wealth"].to_list()[0]
        )
        mask1 = (
            agents._agentsets[1].df["wealth"]
            > agents._agentsets[1].df["wealth"].to_list()[0]
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
