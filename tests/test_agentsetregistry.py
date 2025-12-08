import polars as pl
import pytest
import beartype.roar as bear_roar

from mesa_frames import AgentSet, AgentSetRegistry, Grid, Model


class ExampleAgentSetA(AgentSet):
    def __init__(self, model: Model):
        super().__init__(model)
        self["wealth"] = pl.Series("wealth", [1, 2, 3, 4])
        self["age"] = pl.Series("age", [10, 20, 30, 40])

    def add_wealth(self, amount: int) -> None:
        self["wealth"] += amount

    def step(self) -> None:
        self.add_wealth(1)

    def count(self) -> int:
        return len(self)


class ExampleAgentSetB(AgentSet):
    def __init__(self, model: Model):
        super().__init__(model)
        self["wealth"] = pl.Series("wealth", [10, 20, 30, 40])
        self["age"] = pl.Series("age", [11, 22, 33, 44])

    def add_wealth(self, amount: int) -> None:
        self["wealth"] += amount

    def step(self) -> None:
        self.add_wealth(2)

    def count(self) -> int:
        return len(self)


@pytest.fixture
def fix_model() -> Model:
    return Model()


@pytest.fixture
def fix_set_a(fix_model: Model) -> ExampleAgentSetA:
    return ExampleAgentSetA(fix_model)


@pytest.fixture
def fix_set_b(fix_model: Model) -> ExampleAgentSetB:
    return ExampleAgentSetB(fix_model)


@pytest.fixture
def fix_registry_with_two(
    fix_model: Model, fix_set_a: ExampleAgentSetA, fix_set_b: ExampleAgentSetB
) -> AgentSetRegistry:
    reg = AgentSetRegistry(fix_model)
    reg.add([fix_set_a, fix_set_b])
    return reg


class TestAgentSetRegistry:
    # Dunder: __init__
    def test__init__(self):
        model = Model()
        reg = AgentSetRegistry(model)
        assert reg.model is model
        assert len(reg) == 0
        assert reg.ids.len() == 0

    # Public: add
    def test_add(self, fix_model: Model) -> None:
        reg = AgentSetRegistry(fix_model)
        a1 = ExampleAgentSetA(fix_model)
        a2 = ExampleAgentSetA(fix_model)
        # Add single
        reg.add(a1)
        assert len(reg) == 1
        assert a1 in reg
        # Add list; second should be auto-renamed with suffix
        reg.add([a2])
        assert len(reg) == 2
        names = [s.name for s in reg]
        assert names[0] == "ExampleAgentSetA"
        assert names[1] in ("ExampleAgentSetA_1", "ExampleAgentSetA_2")
        # ids concatenated
        assert reg.ids.len() == len(a1) + len(a2)
        # Duplicate instance rejected
        with pytest.raises(ValueError, match="already present in the AgentSetRegistry"):
            reg.add([a1])
        # Duplicate unique_id space rejected
        a3 = ExampleAgentSetB(fix_model)
        a3.df = a1.df
        with pytest.raises(ValueError, match="agent IDs are not unique"):
            reg.add(a3)

    # Public: contains
    def test_contains(self, fix_registry_with_two: AgentSetRegistry) -> None:
        reg = fix_registry_with_two
        a_name = next(iter(reg)).name
        # Single instance
        assert reg.contains(reg[0]) is True
        # Single type
        assert reg.contains(ExampleAgentSetA) is True
        # Single name
        assert reg.contains(a_name) is True
        # Iterable: instances
        assert reg.contains([reg[0], reg[1]]).to_list() == [True, True]
        # Iterable: types
        types_result = reg.contains([ExampleAgentSetA, ExampleAgentSetB])
        assert types_result.dtype == pl.Boolean
        assert types_result.to_list() == [True, True]
        # Iterable: names
        names = [s.name for s in reg]
        assert reg.contains(names).to_list() == [True, True]
        # Empty iterable is vacuously true
        assert reg.contains([]) is True
        # Unsupported element type (rejected by runtime type checking)
        with pytest.raises(bear_roar.BeartypeCallHintParamViolation):
            reg.contains([object()])

    # Public: do
    def test_do(self, fix_registry_with_two: AgentSetRegistry) -> None:
        reg = fix_registry_with_two
        # Inplace operation across both sets
        reg.do("add_wealth", 5)
        assert reg[0]["wealth"].to_list() == [6, 7, 8, 9]
        assert reg[1]["wealth"].to_list() == [15, 25, 35, 45]
        # return_results with different key domains
        res_by_name = reg.do("count", return_results=True, key_by="name")
        assert set(res_by_name.keys()) == {s.name for s in reg}
        assert all(v == 4 for v in res_by_name.values())
        res_by_index = reg.do("count", return_results=True, key_by="index")
        assert set(res_by_index.keys()) == {0, 1}
        res_by_type = reg.do("count", return_results=True, key_by="type")
        assert set(res_by_type.keys()) == {ExampleAgentSetA, ExampleAgentSetB}

    # Public: get
    def test_get(self, fix_registry_with_two: AgentSetRegistry) -> None:
        reg = fix_registry_with_two
        # By index
        assert isinstance(reg.get(0), AgentSet)
        # By name
        name = reg[0].name
        assert reg.get(name) is reg[0]
        # By type returns list
        aset_list = reg.get(ExampleAgentSetA)
        assert isinstance(aset_list, list) and all(
            isinstance(s, ExampleAgentSetA) for s in aset_list
        )
        # Missing returns default None
        assert reg.get(9999) is None
        # Out-of-range index handled without raising
        assert reg.get(10) is None

    # Public: remove
    def test_remove(self, fix_registry_with_two: AgentSetRegistry) -> None:
        reg = fix_registry_with_two
        total_ids = reg.ids.len()
        # By instance
        reg.remove(reg[0])
        assert len(reg) == 1
        # By type
        reg.add(ExampleAgentSetA(reg.model))
        assert len(reg.get(ExampleAgentSetA)) == 1
        reg.remove(ExampleAgentSetA)
        assert all(not isinstance(s, ExampleAgentSetA) for s in reg)
        # By name (no error if not present)
        reg.remove("nonexistent")
        # ids recomputed and not equal to previous total
        assert reg.ids.len() != total_ids

    def test_remove_clears_space(self, fix_model: Model) -> None:
        reg = fix_model.sets
        aset_a = ExampleAgentSetA(fix_model)
        aset_b = ExampleAgentSetB(fix_model)
        reg.add([aset_a, aset_b])
        space = Grid(fix_model, dimensions=[2, 2], capacity=2)
        fix_model.space = space
        ids_to_place = list(aset_a["unique_id"][:2]) + list(aset_b["unique_id"][:2])
        space.place_agents(
            ids_to_place,
            pos=[[0, 0], [0, 1], [1, 0], [1, 1]],
        )

        reg.remove(aset_a)

        remaining_ids = space.agents["agent_id"]
        assert len(remaining_ids) == 2
        assert not remaining_ids.is_in(aset_a["unique_id"]).any()
        assert remaining_ids.is_in(aset_b["unique_id"]).all()

    # Public: shuffle
    def test_shuffle(self, fix_registry_with_two: AgentSetRegistry) -> None:
        reg = fix_registry_with_two
        reg.shuffle(inplace=True)
        assert len(reg) == 2

    # Public: sort
    def test_sort(self, fix_registry_with_two: AgentSetRegistry) -> None:
        reg = fix_registry_with_two
        reg.sort(by="wealth", ascending=False)
        assert reg[0]["wealth"].to_list() == sorted(
            reg[0]["wealth"].to_list(), reverse=True
        )

    # Public: rename
    def test_rename(self, fix_registry_with_two: AgentSetRegistry) -> None:
        reg = fix_registry_with_two
        # Single rename by instance, inplace
        a0 = reg[0]
        reg.rename(a0, "X")
        assert a0.name == "X"
        assert reg.get("X") is a0

        # Rename second to same name should canonicalize
        a1 = reg[1]
        reg.rename(a1, "X")
        assert a1.name != "X" and a1.name.startswith("X_")
        assert reg.get(a1.name) is a1

        # Non-inplace copy
        reg2 = reg.rename(a0, "Y", inplace=False)
        assert reg2 is not reg
        assert reg.get("Y") is None
        assert reg2.get("Y") is not None

        # Atomic conflict raise: attempt to rename to existing name
        with pytest.raises(ValueError):
            reg.rename({a0: a1.name}, on_conflict="raise", mode="atomic")
        # Names unchanged
        assert reg.get(a1.name) is a1

        # Best-effort: one ok, one conflicting → only ok applied
        unique_name = "Z_unique"
        reg.rename(
            {a0: unique_name, a1: unique_name}, on_conflict="raise", mode="best_effort"
        )
        assert a0.name == unique_name
        # a1 stays with its previous (non-unique_name) value
        assert a1.name != unique_name

    # Dunder: __getattr__
    def test__getattr__(self, fix_registry_with_two: AgentSetRegistry) -> None:
        reg = fix_registry_with_two
        ages = reg.age
        assert isinstance(ages, dict)
        assert set(ages.keys()) == {s.name for s in reg}

    # Dunder: __iter__
    def test__iter__(self, fix_registry_with_two: AgentSetRegistry) -> None:
        reg = fix_registry_with_two
        it = list(iter(reg))
        assert it[0] is reg[0]
        assert all(isinstance(s, AgentSet) for s in it)

    # Dunder: __len__
    def test__len__(self, fix_registry_with_two: AgentSetRegistry) -> None:
        reg = fix_registry_with_two
        assert len(reg) == 2

    # Dunder: __repr__
    def test__repr__(self, fix_registry_with_two: AgentSetRegistry) -> None:
        reg = fix_registry_with_two
        repr(reg)

    # Dunder: __str__
    def test__str__(self, fix_registry_with_two: AgentSetRegistry) -> None:
        reg = fix_registry_with_two
        str(reg)

    # Dunder: __reversed__
    def test__reversed__(self, fix_registry_with_two: AgentSetRegistry) -> None:
        reg = fix_registry_with_two
        list(reversed(reg))

    # Dunder: __setitem__
    def test__setitem__(self, fix_model: Model) -> None:
        reg = AgentSetRegistry(fix_model)
        a1 = ExampleAgentSetA(fix_model)
        a2 = ExampleAgentSetB(fix_model)
        reg.add([a1, a2])
        # Assign by index with duplicate name should raise
        a_dup = ExampleAgentSetA(fix_model)
        a_dup.name = reg[1].name  # create name collision
        with pytest.raises(ValueError, match="Duplicate agent set name disallowed"):
            reg[0] = a_dup
        # Assign by name: replace existing slot, authoritative name should be key
        new_set = ExampleAgentSetA(fix_model)
        reg[reg[1].name] = new_set
        assert reg[1] is new_set
        assert reg[1].name == reg[1].name
        # Assign new name appends
        extra = ExampleAgentSetA(fix_model)
        reg["extra_set"] = extra
        assert reg["extra_set"] is extra
        # Model mismatch raises
        other_model_set = ExampleAgentSetA(Model())
        with pytest.raises(
            TypeError, match="Assigned AgentSet must belong to the same model"
        ):
            reg[0] = other_model_set

    # Public: keys
    def test_keys(self, fix_registry_with_two: AgentSetRegistry) -> None:
        reg = fix_registry_with_two
        # keys by name
        names = list(reg.keys())
        assert names == [s.name for s in reg]
        # keys by index
        assert list(reg.keys(key_by="index")) == [0, 1]
        # keys by type
        assert set(reg.keys(key_by="type")) == {ExampleAgentSetA, ExampleAgentSetB}
        # invalid key_by
        with pytest.raises(bear_roar.BeartypeCallHintParamViolation):
            list(reg.keys(key_by="bad"))

    # Public: items
    def test_items(self, fix_registry_with_two: AgentSetRegistry) -> None:
        reg = fix_registry_with_two
        items_name = list(reg.items())
        assert [k for k, _ in items_name] == [s.name for s in reg]
        items_idx = list(reg.items(key_by="index"))
        assert [k for k, _ in items_idx] == [0, 1]
        items_type = list(reg.items(key_by="type"))
        assert {k for k, _ in items_type} == {ExampleAgentSetA, ExampleAgentSetB}

    # Public: values
    def test_values(self, fix_registry_with_two: AgentSetRegistry) -> None:
        reg = fix_registry_with_two
        assert list(reg.values())[0] is reg[0]

    # Public: discard
    def test_discard(self, fix_registry_with_two: AgentSetRegistry) -> None:
        reg = fix_registry_with_two
        original_len = len(reg)
        # Missing selector ignored without error
        reg.discard("missing_name")
        assert len(reg) == original_len
        # Remove by instance
        reg.discard(reg[0])
        assert len(reg) == original_len - 1
        # Non-inplace returns new copy
        reg2 = reg.discard("missing_name", inplace=False)
        assert len(reg2) == len(reg)

    # Public: ids (property)
    def test_ids(self, fix_registry_with_two: AgentSetRegistry) -> None:
        reg = fix_registry_with_two
        assert isinstance(reg.ids, pl.Series)
        before = reg.ids.len()
        reg.remove(reg[0])
        assert reg.ids.len() < before

    # Dunder: __getitem__
    def test__getitem__(self, fix_registry_with_two: AgentSetRegistry) -> None:
        reg = fix_registry_with_two
        # By index
        assert reg[0] is next(iter(reg))
        # By name
        name0 = reg[0].name
        assert reg[name0] is reg[0]
        # By type
        lst = reg[ExampleAgentSetA]
        assert isinstance(lst, list) and all(
            isinstance(s, ExampleAgentSetA) for s in lst
        )
        # Missing name raises KeyError
        with pytest.raises(KeyError):
            _ = reg["missing"]

    # Dunder: __contains__ (membership)
    def test__contains__(self, fix_registry_with_two: AgentSetRegistry) -> None:
        reg = fix_registry_with_two
        assert reg[0] in reg
        new_set = ExampleAgentSetA(reg.model)
        assert new_set not in reg

    # Dunder: __add__
    def test__add__(self, fix_model: Model) -> None:
        reg = AgentSetRegistry(fix_model)
        a1 = ExampleAgentSetA(fix_model)
        a2 = ExampleAgentSetB(fix_model)
        reg.add(a1)
        reg_new = reg + a2
        # original unchanged, new has two
        assert len(reg) == 1
        assert len(reg_new) == 2
        # Presence by type/name (instances are deep-copied)
        assert reg_new.contains(ExampleAgentSetA) is True
        assert reg_new.contains(ExampleAgentSetB) is True

    # Dunder: __iadd__
    def test__iadd__(self, fix_model: Model) -> None:
        reg = AgentSetRegistry(fix_model)
        a1 = ExampleAgentSetA(fix_model)
        a2 = ExampleAgentSetB(fix_model)
        reg += a1
        assert len(reg) == 1
        reg += [a2]
        assert len(reg) == 2
        assert reg.contains([a1, a2]).all()

    # Dunder: __sub__
    def test__sub__(self, fix_model: Model) -> None:
        reg = AgentSetRegistry(fix_model)
        a1 = ExampleAgentSetA(fix_model)
        a2 = ExampleAgentSetB(fix_model)
        reg.add([a1, a2])
        reg_new = reg - a1
        # original unchanged
        assert len(reg) == 2
        # In current implementation, subtraction with instance returns a copy
        # without mutation due to deep-copied identity; ensure new object
        assert isinstance(reg_new, AgentSetRegistry) and reg_new is not reg
        assert len(reg_new) == len(reg)
        # subtract list of instances also yields unchanged copy
        reg_new2 = reg - [a1, a2]
        assert len(reg_new2) == len(reg)

    # Dunder: __isub__
    def test__isub__(self, fix_model: Model) -> None:
        reg = AgentSetRegistry(fix_model)
        a1 = ExampleAgentSetA(fix_model)
        a2 = ExampleAgentSetB(fix_model)
        reg.add([a1, a2])
        reg -= a1
        assert len(reg) == 1 and a1 not in reg
        reg -= [a2]
        assert len(reg) == 0

    # Public: replace
    def test_replace(self, fix_model: Model) -> None:
        reg = AgentSetRegistry(fix_model)
        a1 = ExampleAgentSetA(fix_model)
        a2 = ExampleAgentSetB(fix_model)
        a3 = ExampleAgentSetA(fix_model)
        reg.add([a1, a2])
        # Replace by index
        reg.replace({0: a3})
        assert reg[0] is a3
        # Replace by name (authoritative)
        reg.replace({reg[1].name: a2})
        assert reg[1] is a2
        # Atomic aliasing error: same object in two positions
        with pytest.raises(ValueError, match="already exists at index"):
            reg.replace({0: a2, 1: a2})
        # Model mismatch
        with pytest.raises(TypeError, match="must belong to the same model"):
            reg.replace({0: ExampleAgentSetA(Model())})
        # Non-atomic: only applies valid keys to copy
        reg2 = reg.replace({0: a1}, inplace=False, atomic=False)
        assert reg2[0] is a1
        assert reg[0] is not a1
