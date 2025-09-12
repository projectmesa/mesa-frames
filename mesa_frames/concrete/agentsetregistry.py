"""
Concrete implementation of the agents collection for mesa-frames.

This module provides the concrete implementation of the agents collection class
for the mesa-frames library. It defines the AgentSetRegistry class, which serves as a
container for all agent sets in a model, leveraging DataFrame-based storage for
improved performance.

Classes:
    AgentSetRegistry(AbstractAgentSetRegistry):
        A collection of AgentSets. This class acts as a container for all
        agents in the model, organizing them into separate AgentSet instances
        based on their types.

The AgentSetRegistry class is designed to be used within Model instances to manage
all agents in the simulation. It provides methods for adding, removing, and
accessing agents and agent sets, while taking advantage of the performance
benefits of DataFrame-based agent storage.

Usage:
    The AgentSetRegistry class is typically instantiated and used within a Model subclass:

    from mesa_frames.concrete.model import Model
    from mesa_frames.concrete.agents import AgentSetRegistry
    from mesa_frames.concrete import AgentSet

    class MyCustomModel(Model):
        def __init__(self):
            super().__init__()
            # Adding agent sets to the collection
            self.sets += AgentSet(self)
            self.sets += AnotherAgentSet(self)

        def step(self):
            # Step all agent sets
            self.sets.do("step")

Note:
    This concrete implementation builds upon the abstract AgentSetRegistry class
    defined in the mesa_frames.abstract package, providing a ready-to-use
    agents collection that integrates with the DataFrame-based agent storage system.

For more detailed information on the AgentSetRegistry class and its methods, refer to
the class docstring.
"""

from __future__ import annotations  # For forward references

from collections.abc import Collection, Iterable, Iterator, Sequence
from typing import Any, Literal, Self, overload, cast

import polars as pl

from mesa_frames.abstract.agentsetregistry import (
    AbstractAgentSetRegistry,
)
from mesa_frames.concrete.agentset import AgentSet
from mesa_frames.types_ import BoolSeries, KeyBy, AgentSetSelector


class AgentSetRegistry(AbstractAgentSetRegistry):
    """A collection of AgentSets. All agents of the model are stored here."""

    _agentsets: list[AgentSet]
    _ids: pl.Series

    def __init__(self, model: mesa_frames.concrete.model.Model) -> None:
        """Initialize a new AgentSetRegistry.

        Parameters
        ----------
        model : mesa_frames.concrete.model.Model
            The model associated with the AgentSetRegistry.
        """
        self._model = model
        self._agentsets = []
        self._ids = pl.Series(name="unique_id", dtype=pl.UInt64)

    def add(
        self,
        sets: AgentSet | Iterable[AgentSet],
        inplace: bool = True,
    ) -> Self:
        obj = self._get_obj(inplace)
        other_list = obj._return_agentsets_list(sets)
        if obj._check_agentsets_presence(other_list).any():
            raise ValueError(
                "Some agentsets are already present in the AgentSetRegistry."
            )
        for agentset in other_list:
            # Set name if not already set, using class name
            if agentset.name is None:
                base_name = agentset.__class__.__name__
                name = obj._generate_name(base_name)
                agentset.name = name
        new_ids = pl.concat(
            [obj._ids] + [pl.Series(agentset["unique_id"]) for agentset in other_list]
        )
        if new_ids.is_duplicated().any():
            raise ValueError("Some of the agent IDs are not unique.")
        obj._agentsets.extend(other_list)
        obj._ids = new_ids
        return obj

    @overload
    def contains(self, sets: AgentSet | type[AgentSet] | str) -> bool: ...

    @overload
    def contains(
        self,
        sets: Iterable[AgentSet] | Iterable[type[AgentSet]] | Iterable[str],
    ) -> pl.Series: ...

    def contains(
        self,
        sets: AgentSet
        | type[AgentSet]
        | str
        | Iterable[AgentSet]
        | Iterable[type[AgentSet]]
        | Iterable[str],
    ) -> bool | pl.Series:
        # Single value fast paths
        if isinstance(sets, AgentSet):
            return self._check_agentsets_presence([sets]).any()
        if isinstance(sets, type) and issubclass(sets, AgentSet):
            return any(isinstance(s, sets) for s in self._agentsets)
        if isinstance(sets, str):
            return any(s.name == sets for s in self._agentsets)

        # Iterable paths without materializing unnecessarily

        if isinstance(sets, Sized) and len(sets) == 0:  # type: ignore[arg-type]
            return True
        it = iter(sets)  # type: ignore[arg-type]
        try:
            first = next(it)
        except StopIteration:
            return True

        if isinstance(first, AgentSet):
            lst = [first, *it]
            return self._check_agentsets_presence(lst)

        if isinstance(first, type) and issubclass(first, AgentSet):
            present_types = {type(s) for s in self._agentsets}

            def has_type(t: type[AgentSet]) -> bool:
                return any(issubclass(pt, t) for pt in present_types)

            return pl.Series(
                (has_type(t) for t in chain([first], it)), dtype=pl.Boolean
            )

        if isinstance(first, str):
            names = {s.name for s in self._agentsets if s.name is not None}
            return pl.Series((x in names for x in chain([first], it)), dtype=pl.Boolean)

        raise TypeError("Unsupported type for contains()")

    @overload
    def do(
        self,
        method_name: str,
        *args: Any,
        sets: AgentSetSelector | None = None,
        return_results: Literal[False] = False,
        inplace: bool = True,
        key_by: KeyBy = "name",
        **kwargs: Any,
    ) -> Self: ...

    @overload
    def do(
        self,
        method_name: str,
        *args: Any,
        sets: AgentSetSelector,
        return_results: Literal[True],
        inplace: bool = True,
        key_by: KeyBy = "name",
        **kwargs: Any,
    ) -> dict[str, Any] | dict[int, Any] | dict[type[AgentSet], Any]: ...

    def do(
        self,
        method_name: str,
        *args: Any,
        sets: AgentSetSelector = None,
        return_results: bool = False,
        inplace: bool = True,
        key_by: KeyBy = "name",
        **kwargs: Any,
    ) -> Self | Any:
        obj = self._get_obj(inplace)
        target_sets = obj._resolve_selector(sets)
        if return_results:

            def make_key(i: int, s: AgentSet) -> Any:
                if key_by == "name":
                    return s.name
                if key_by == "index":
                    return i
                if key_by == "type":
                    return type(s)
                return s  # backward-compatible: key by object

            return {
                make_key(i, s): s.do(
                    method_name, *args, return_results=True, inplace=inplace, **kwargs
                )
                for i, s in enumerate(target_sets)
            }
        obj._agentsets = [
            s.do(method_name, *args, return_results=False, inplace=inplace, **kwargs)
            for s in target_sets
        ]
        return obj

    @overload
    def get(self, key: int, default: None = ...) -> AgentSet | None: ...

    @overload
    def get(self, key: str, default: None = ...) -> AgentSet | None: ...

    @overload
    def get(self, key: type[AgentSet], default: None = ...) -> list[AgentSet]: ...

    @overload
    def get(
        self,
        key: int | str | type[AgentSet],
        default: AgentSet | list[AgentSet] | None,
    ) -> AgentSet | list[AgentSet] | None: ...

    def get(
        self,
        key: int | str | type[AgentSet],
        default: AgentSet | list[AgentSet] | None = None,
    ) -> AgentSet | list[AgentSet] | None:
        try:
            if isinstance(key, int):
                return self._agentsets[key]
            if isinstance(key, str):
                for s in self._agentsets:
                    if s.name == key:
                        return s
                return default
            if isinstance(key, type) and issubclass(key, AgentSet):
                return [s for s in self._agentsets if isinstance(s, key)]
        except (IndexError, KeyError, TypeError):
            return default
        return default

    def remove(
        self,
        sets: AgentSetSelector,
        inplace: bool = True,
    ) -> Self:
        obj = self._get_obj(inplace)
        if agents is None or (isinstance(agents, Iterable) and len(agents) == 0):
            return obj
        if isinstance(agents, AgentSet):
            agents = [agents]
        if isinstance(agents, Iterable) and isinstance(next(iter(agents)), AgentSet):
            # We have to get the index of the original AgentSet because the copy made AgentSets with different hash
            ids = [self._agentsets.index(agentset) for agentset in iter(agents)]
            ids.sort(reverse=True)
            removed_ids = pl.Series(dtype=pl.UInt64)
            for id in ids:
                removed_ids = pl.concat(
                    [
                        removed_ids,
                        pl.Series(obj._agentsets[id]["unique_id"], dtype=pl.UInt64),
                    ]
                )
                obj._agentsets.pop(id)

        else:  # IDsLike
            if isinstance(agents, (int, np.uint64)):
                agents = [agents]
            elif isinstance(agents, DataFrame):
                agents = agents["unique_id"]
            removed_ids = pl.Series(agents, dtype=pl.UInt64)
            deleted = 0

            for agentset in obj._agentsets:
                initial_len = len(agentset)
                agentset._discard(removed_ids)
                deleted += initial_len - len(agentset)
                if deleted == len(removed_ids):
                    break
            if deleted < len(removed_ids):  # TODO: fix type hint
                raise KeyError(
                    "There exist some IDs which are not present in any agentset"
                )
        try:
            obj.space.remove_agents(removed_ids, inplace=True)
        except ValueError:
            pass
        obj._ids = obj._ids.filter(obj._ids.is_in(removed_ids).not_())
        return obj

    def select(
        self,
        mask: AgnosticAgentMask | IdsLike | dict[AgentSet, AgentMask] = None,
        filter_func: Callable[[AgentSet], AgentMask] | None = None,
        n: int | None = None,
        inplace: bool = True,
        negate: bool = False,
    ) -> Self:
        obj = self._get_obj(inplace)
        agentsets_masks = obj._get_bool_masks(mask)
        if n is not None:
            n = n // len(agentsets_masks)
        obj._agentsets = [
            agentset.select(
                mask=mask, filter_func=filter_func, n=n, negate=negate, inplace=inplace
            )
            for agentset, mask in agentsets_masks.items()
        ]
        return obj

    def set(
        self,
        attr_names: str | dict[AgentSet, Any] | Collection[str],
        values: Any | None = None,
        mask: AgnosticAgentMask | IdsLike | dict[AgentSet, AgentMask] = None,
        inplace: bool = True,
    ) -> Self:
        obj = self._get_obj(inplace)
        agentsets_masks = obj._get_bool_masks(mask)
        if isinstance(attr_names, dict):
            for agentset, values in attr_names.items():
                if not inplace:
                    # We have to get the index of the original AgentSet because the copy made AgentSets with different hash
                    id = self._agentsets.index(agentset)
                    agentset = obj._agentsets[id]
                agentset.set(
                    attr_names=values, mask=agentsets_masks[agentset], inplace=True
                )
        else:
            obj._agentsets = [
                agentset.set(
                    attr_names=attr_names, values=values, mask=mask, inplace=True
                )
                for agentset, mask in agentsets_masks.items()
            ]
        return obj

    def shuffle(self, inplace: bool = True) -> Self:
    def shuffle(self, inplace: bool = False) -> Self:
        obj = self._get_obj(inplace)
        obj._agentsets = [agentset.shuffle(inplace=True) for agentset in obj._agentsets]
        return obj

    def sort(
        self,
        by: str | Sequence[str],
        ascending: bool | Sequence[bool] = True,
        inplace: bool = True,
        **kwargs: Any,
    ) -> Self:
        obj = self._get_obj(inplace)
        obj._agentsets = [
            agentset.sort(by=by, ascending=ascending, inplace=inplace, **kwargs)
            for agentset in obj._agentsets
        ]
        return obj

    def _check_ids_presence(self, other: list[AgentSet]) -> pl.DataFrame:
        """Check if the IDs of the agents to be added are unique.

        Parameters
        ----------
        other : list[AgentSet]
            The AgentSets to check.

        Returns
        -------
        pl.DataFrame
            A DataFrame with the unique IDs and a boolean column indicating if they are present.
        """
        presence_df = pl.DataFrame(
            data={"unique_id": self._ids, "present": True},
            schema={"unique_id": pl.UInt64, "present": pl.Boolean},
        )
        for agentset in other:
            new_ids = pl.Series(agentset.index, dtype=pl.UInt64)
            presence_df = pl.concat(
                [
                    presence_df,
                    (
                        new_ids.is_in(presence_df["unique_id"])
                        .to_frame("present")
                        .with_columns(unique_id=new_ids)
                        .select(["unique_id", "present"])
                    ),
                ]
            )
        presence_df = presence_df.slice(self._ids.len())
        return presence_df

    def _check_agentsets_presence(self, other: list[AgentSet]) -> pl.Series:
        """Check if the agent sets to be added are already present in the AgentSetRegistry.

        Parameters
        ----------
        other : list[AgentSet]
            The AgentSets to check.

        Returns
        -------
        pl.Series
            A boolean Series indicating if the agent sets are present.

        Raises
        ------
        ValueError
            If the agent sets are already present in the AgentSetRegistry.
        """
        other_set = set(other)
        return pl.Series(
            [agentset in other_set for agentset in self._agentsets], dtype=pl.Boolean
        )

    def _resolve_selector(self, selector: AgentSetSelector = None) -> list[AgentSet]:
        """Resolve a selector (instance/type/name or collection) to a list of AgentSets."""
        if selector is None:
            return list(self._agentsets)
        # Single instance
        if isinstance(selector, AgentSet):
            return [selector] if selector in self._agentsets else []
        # Single type
        if isinstance(selector, type) and issubclass(selector, AgentSet):
            return [s for s in self._agentsets if isinstance(s, selector)]
        # Single name
        if isinstance(selector, str):
            return [s for s in self._agentsets if s.name == selector]
        # Collection of mixed selectors
        selected: list[AgentSet] = []
        for item in selector:  # type: ignore[assignment]
            if isinstance(item, AgentSet):
                if item in self._agentsets:
                    selected.append(item)
            elif isinstance(item, type) and issubclass(item, AgentSet):
                selected.extend([s for s in self._agentsets if isinstance(s, item)])
            elif isinstance(item, str):
                selected.extend([s for s in self._agentsets if s.name == item])
            else:
                raise TypeError("Unsupported selector element type")
        # Deduplicate while preserving order
        seen = set()
        result = []
        for s in selected:
            if s not in seen:
                seen.add(s)
                result.append(s)
        return result

    def _return_agentsets_list(
        self, agentsets: AgentSet | Iterable[AgentSet]
    ) -> list[AgentSet]:
        """Convert the agentsets to a list of AgentSet.

        Parameters
        ----------
        agentsets : AgentSet | Iterable[AgentSet]

        Returns
        -------
        list[AgentSet]
        """
        return [agentsets] if isinstance(agentsets, AgentSet) else list(agentsets)

    def _generate_name(self, base_name: str) -> str:
        """Generate a unique name for an agent set."""
        existing_names = [
            agentset.name for agentset in self._agentsets if agentset.name is not None
        ]
        if base_name not in existing_names:
            return base_name
        counter = 1
        candidate = f"{base_name}_{counter}"
        while candidate in existing_names:
            counter += 1
            candidate = f"{base_name}_{counter}"
        return candidate

    def __getattr__(self, name: str) -> Any | dict[str, Any]:
        # Avoids infinite recursion of private attributes
        if name.startswith("_"):
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )
        # Delegate attribute access to sets; map results by set name
        return {cast(str, s.name): getattr(s, name) for s in self._agentsets}

    def __iter__(self) -> Iterator[AgentSet]:
        return iter(self._agentsets)

    def __len__(self) -> int:
        return len(self._agentsets)

    def __repr__(self) -> str:
        return "\n".join([repr(agentset) for agentset in self._agentsets])

    def __reversed__(self) -> Iterator[AgentSet]:
        return reversed(self._agentsets)

    def __setitem__(self, key: int | str, value: AgentSet) -> None:
        """Assign/replace a single AgentSet at an index or name.

        Enforces name uniqueness and model consistency.
        """
        if value.model is not self.model:
            raise TypeError("Assigned AgentSet must belong to the same model")
        if isinstance(key, int):
            if value.name is not None:
                for i, s in enumerate(self._agentsets):
                    if i != key and s.name == value.name:
                        raise ValueError(
                            f"Duplicate agent set name disallowed: {value.name}"
                        )
            self._agentsets[key] = value
        elif isinstance(key, str):
            try:
                value.rename(key)
            except Exception:
                if hasattr(value, "_name"):
                    setattr(value, "_name", key)
            idx = None
            for i, s in enumerate(self._agentsets):
                if s.name == key:
                    idx = i
                    break
            if idx is None:
                self._agentsets.append(value)
            else:
                self._agentsets[idx] = value
        else:
            raise TypeError("Key must be int index or str name")
        # Recompute ids cache
        if self._agentsets:
            self._ids = pl.concat(
                [pl.Series(name="unique_id", dtype=pl.UInt64)]
                + [pl.Series(s["unique_id"]) for s in self._agentsets]
            )
        else:
            self._ids = pl.Series(name="unique_id", dtype=pl.UInt64)

    def __str__(self) -> str:
        return "\n".join([str(agentset) for agentset in self._agentsets])

    def keys(self, *, key_by: KeyBy = "name") -> Iterable[Any]:
        if key_by not in ("name", "index", "type"):
            raise ValueError("key_by must be 'name'|'index'|'type'")
        if key_by == "index":
            for i in range(len(self._agentsets)):
                yield i
            return
        if key_by == "type":
            for s in self._agentsets:
                yield type(s)
            return
        # name
        for s in self._agentsets:
            if s.name is not None:
                yield s.name

    def items(self, *, key_by: KeyBy = "name") -> Iterable[tuple[Any, AgentSet]]:
        if key_by not in ("name", "index", "type"):
            raise ValueError("key_by must be 'name'|'index'|'type'")
        if key_by == "index":
            for i, s in enumerate(self._agentsets):
                yield i, s
            return
        if key_by == "type":
            for s in self._agentsets:
                yield type(s), s
            return
        # name
        for s in self._agentsets:
            if s.name is not None:
                yield s.name, s

    def values(self) -> Iterable[AgentSet]:
        return iter(self._agentsets)

    @overload
    def __getitem__(self, key: int) -> AgentSet: ...

    @overload
    def __getitem__(self, key: str) -> AgentSet: ...

    @overload
    def __getitem__(self, key: type[AgentSet]) -> list[AgentSet]: ...

    def __getitem__(self, key: int | str | type[AgentSet]) -> AgentSet | list[AgentSet]:
        """Retrieve AgentSet(s) by index, name, or type."""
        if isinstance(key, int):
            return self._agentsets[key]
        if isinstance(key, str):
            for s in self._agentsets:
                if s.name == key:
                    return s
            raise KeyError(f"Agent set '{key}' not found")
        if isinstance(key, type) and issubclass(key, AgentSet):
            return [s for s in self._agentsets if isinstance(s, key)]
        raise TypeError("Key must be int, str (name), or AgentSet type")
