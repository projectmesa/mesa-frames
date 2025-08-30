from copy import copy, deepcopy

import pytest

from mesa_frames import AgentsDF, ModelDF
from tests.test_agentset import (
    ExampleAgentSetPolars,
    fix1_AgentSetPolars,
    fix2_AgentSetPolars,
)
from tests.test_agents import fix_AgentsDF


class TestAgentSetsAccessor:
    def test___getitem__(self, fix_AgentsDF):
        agents = fix_AgentsDF
        s1 = agents.sets[0]
        s2 = agents.sets[1]
        # int
        assert agents.sets[0] is s1
        assert agents.sets[1] is s2
        with pytest.raises(IndexError):
            _ = agents.sets[2]
        # str
        assert agents.sets[s1.name] is s1
        assert agents.sets[s2.name] is s2
        with pytest.raises(KeyError):
            _ = agents.sets["__missing__"]
        # type → always list
        lst = agents.sets[ExampleAgentSetPolars]
        assert isinstance(lst, list)
        assert s1 in lst and s2 in lst and len(lst) == 2

    def test_get(self, fix_AgentsDF):
        agents = fix_AgentsDF
        assert agents.sets.get("__missing__") is None
        assert agents.sets.get(999, default="x") == "x"

        class Temp(ExampleAgentSetPolars):
            pass

        assert agents.sets.get(Temp) == []
        assert agents.sets.get(Temp, default=None) == []
        assert agents.sets.get(Temp, default=["fallback"]) == ["fallback"]

    def test_first(self, fix_AgentsDF):
        agents = fix_AgentsDF
        assert agents.sets.first(ExampleAgentSetPolars) is agents.sets[0]

        class Temp(ExampleAgentSetPolars):
            pass

        with pytest.raises(KeyError):
            agents.sets.first(Temp)

    def test_all(self, fix_AgentsDF):
        agents = fix_AgentsDF
        assert agents.sets.all(ExampleAgentSetPolars) == [
            agents.sets[0],
            agents.sets[1],
        ]

        class Temp(ExampleAgentSetPolars):
            pass

        assert agents.sets.all(Temp) == []

    def test_at(self, fix_AgentsDF):
        agents = fix_AgentsDF
        assert agents.sets.at(0) is agents.sets[0]
        assert agents.sets.at(1) is agents.sets[1]

    def test_keys(self, fix_AgentsDF):
        agents = fix_AgentsDF
        s1 = agents.sets[0]
        s2 = agents.sets[1]
        assert list(agents.sets.keys(key_by="index")) == [0, 1]
        assert list(agents.sets.keys(key_by="object")) == [s1, s2]
        assert list(agents.sets.keys(key_by="name")) == [s1.name, s2.name]
        assert list(agents.sets.keys(key_by="type")) == [type(s1), type(s2)]

    def test_items(self, fix_AgentsDF):
        agents = fix_AgentsDF
        s1 = agents.sets[0]
        s2 = agents.sets[1]
        assert list(agents.sets.items(key_by="index")) == [(0, s1), (1, s2)]
        assert list(agents.sets.items(key_by="object")) == [(s1, s1), (s2, s2)]

    def test_values(self, fix_AgentsDF):
        agents = fix_AgentsDF
        s1 = agents.sets[0]
        s2 = agents.sets[1]
        assert list(agents.sets.values()) == [s1, s2]

    def test_iter(self, fix_AgentsDF):
        agents = fix_AgentsDF
        s1 = agents.sets[0]
        s2 = agents.sets[1]
        assert list(agents.sets.iter(key_by="name")) == [(s1.name, s1), (s2.name, s2)]

    def test_mapping(self, fix_AgentsDF):
        agents = fix_AgentsDF
        s1 = agents.sets[0]
        s2 = agents.sets[1]
        by_type_map = agents.sets.mapping(key_by="type")
        assert list(by_type_map.keys()) == [type(s1)]
        assert by_type_map[type(s1)] is s2

    def test_by_name(self, fix_AgentsDF):
        agents = fix_AgentsDF
        s1 = agents.sets[0]
        s2 = agents.sets[1]
        name_map = agents.sets.by_name
        assert name_map[s1.name] is s1
        assert name_map[s2.name] is s2
        with pytest.raises(TypeError):
            name_map["X"] = s1  # type: ignore[index]

    def test_by_type(self, fix_AgentsDF):
        agents = fix_AgentsDF
        s1 = agents.sets[0]
        s2 = agents.sets[1]
        grouped = agents.sets.by_type
        assert list(grouped.keys()) == [type(s1)]
        assert grouped[type(s1)] == [s1, s2]

    def test___contains__(self, fix_AgentsDF):
        agents = fix_AgentsDF
        s1 = agents.sets[0]
        s2 = agents.sets[1]
        assert s1.name in agents.sets
        assert s2.name in agents.sets
        assert s1 in agents.sets and s2 in agents.sets

    def test___len__(self, fix_AgentsDF):
        agents = fix_AgentsDF
        assert len(agents.sets) == 2

    def test___iter__(self, fix_AgentsDF):
        agents = fix_AgentsDF
        s1 = agents.sets[0]
        s2 = agents.sets[1]
        assert list(iter(agents.sets)) == [s1, s2]

    def test_copy_and_deepcopy_rebinds_accessor(self, fix_AgentsDF):
        agents = fix_AgentsDF
        s1 = agents.sets[0]
        s2 = agents.sets[1]
        a2 = copy(agents)
        acc2 = a2.sets  # lazily created
        assert acc2._parent is a2
        assert acc2 is not agents.sets
        a3 = deepcopy(agents)
        acc3 = a3.sets  # lazily created
        assert acc3._parent is a3
        assert acc3 is not agents.sets and acc3 is not acc2
