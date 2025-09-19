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

<<<<<<< HEAD
from collections.abc import Collection, Iterable, Iterator, Sequence
from typing import Any, Literal, Self, overload, cast
from collections.abc import Sized
from itertools import chain
=======
from collections import defaultdict
from collections.abc import Callable, Collection, Iterable, Iterator, Sequence
from typing import Any, Literal, Self, cast, overload

import numpy as np
>>>>>>> 51c54cd666d876a5debb1b7dd71556ee9c458956
import polars as pl

from mesa_frames.abstract.agentsetregistry import (
    AbstractAgentSetRegistry,
)
from mesa_frames.concrete.agentset import AgentSet
<<<<<<< HEAD
from mesa_frames.types_ import BoolSeries, KeyBy, AgentSetSelector
=======
from mesa_frames.types_ import (
    AgentMask,
    AgnosticAgentMask,
    BoolSeries,
    DataFrame,
    IdsLike,
    Index,
    Series,
)
>>>>>>> 51c54cd666d876a5debb1b7dd71556ee9c458956


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
<<<<<<< HEAD
        sets: AgentSet | Iterable[AgentSet],
        inplace: bool = True,
    ) -> Self:
        obj = self._get_obj(inplace)
        other_list = obj._return_agentsets_list(sets)
=======
        agents: AgentSet | Iterable[AgentSet],
        inplace: bool = True,
    ) -> Self:
        """Add an AgentSet to the AgentSetRegistry.

        Parameters
        ----------
        agents : AgentSet | Iterable[AgentSet]
            The AgentSets to add.
        inplace : bool, optional
            Whether to add the AgentSets in place. Defaults to True.

        Returns
        -------
        Self
            The updated AgentSetRegistry.

        Raises
        ------
        ValueError
            If any AgentSets are already present or if IDs are not unique.
        """
        obj = self._get_obj(inplace)
        other_list = obj._return_agentsets_list(agents)
>>>>>>> 51c54cd666d876a5debb1b7dd71556ee9c458956
        if obj._check_agentsets_presence(other_list).any():
            raise ValueError(
                "Some agentsets are already present in the AgentSetRegistry."
            )
<<<<<<< HEAD
        # Ensure unique names across existing and to-be-added sets
        existing_names = {s.name for s in obj._agentsets}
        for agentset in other_list:
            base_name = agentset.name or agentset.__class__.__name__
            name = base_name
            if name in existing_names:
                counter = 1
                candidate = f"{base_name}_{counter}"
                while candidate in existing_names:
                    counter += 1
                    candidate = f"{base_name}_{counter}"
                name = candidate
            # Assign back if changed or was None
            if name != (agentset.name or base_name):
                agentset.name = name
            existing_names.add(name)
=======
>>>>>>> 51c54cd666d876a5debb1b7dd71556ee9c458956
        new_ids = pl.concat(
            [obj._ids] + [pl.Series(agentset["unique_id"]) for agentset in other_list]
        )
        if new_ids.is_duplicated().any():
            raise ValueError("Some of the agent IDs are not unique.")
        obj._agentsets.extend(other_list)
        obj._ids = new_ids
        return obj

<<<<<<< HEAD
    def rename(
        self,
        target: (
            AgentSet
            | str
            | dict[AgentSet | str, str]
            | list[tuple[AgentSet | str, str]]
        ),
        new_name: str | None = None,
        *,
        on_conflict: Literal["canonicalize", "raise"] = "canonicalize",
        mode: Literal["atomic", "best_effort"] = "atomic",
        inplace: bool = True,
    ) -> Self:
        """Rename AgentSets with conflict handling.

        Supports single-target ``(set | old_name, new_name)`` and batch rename via
        dict or list of pairs. Names remain unique across the registry.
        """

        # Normalize to list of (index_in_self, desired_name) using the original registry
        def _resolve_one(x: AgentSet | str) -> int:
            if isinstance(x, AgentSet):
                for i, s in enumerate(self._agentsets):
                    if s is x:
                        return i
                raise KeyError("AgentSet not found in registry")
            # name lookup on original registry
            for i, s in enumerate(self._agentsets):
                if s.name == x:
                    return i
            raise KeyError(f"Agent set '{x}' not found")

        if isinstance(target, (AgentSet, str)):
            if new_name is None:
                raise TypeError("new_name must be provided for single rename")
            pairs_idx: list[tuple[int, str]] = [(_resolve_one(target), new_name)]
            single = True
        elif isinstance(target, dict):
            pairs_idx = [(_resolve_one(k), v) for k, v in target.items()]
            single = False
        else:
            pairs_idx = [(_resolve_one(k), v) for k, v in target]
            single = False

        # Choose object to mutate
        obj = self._get_obj(inplace)
        # Translate indices to object AgentSets in the selected registry object
        target_sets = [obj._agentsets[i] for i, _ in pairs_idx]

        # Build the set of names that remain fixed (exclude targets' current names)
        targets_set = set(target_sets)
        fixed_names: set[str] = {
            s.name
            for s in obj._agentsets
            if s.name is not None and s not in targets_set
        }  # type: ignore[comparison-overlap]

        # Plan final names
        final: list[tuple[AgentSet, str]] = []
        used = set(fixed_names)

        def _canonicalize(base: str) -> str:
            if base not in used:
                used.add(base)
                return base
            counter = 1
            cand = f"{base}_{counter}"
            while cand in used:
                counter += 1
                cand = f"{base}_{counter}"
            used.add(cand)
            return cand

        errors: list[Exception] = []
        for aset, (_idx, desired) in zip(target_sets, pairs_idx):
            if on_conflict == "canonicalize":
                final_name = _canonicalize(desired)
                final.append((aset, final_name))
            else:  # on_conflict == 'raise'
                if desired in used:
                    err = ValueError(
                        f"Duplicate agent set name disallowed: '{desired}'"
                    )
                    if mode == "atomic":
                        errors.append(err)
                    else:
                        # best_effort: skip this rename
                        continue
                else:
                    used.add(desired)
                    final.append((aset, desired))

        if errors and mode == "atomic":
            # Surface first meaningful error
            raise errors[0]

        # Apply renames
        for aset, newn in final:
            # Set the private name directly to avoid external uniqueness hooks
            if hasattr(aset, "_name"):
                aset._name = newn  # type: ignore[attr-defined]

        return obj

    def replace(
        self,
        mapping: (dict[int | str, AgentSet] | list[tuple[int | str, AgentSet]]),
        *,
        inplace: bool = True,
        atomic: bool = True,
    ) -> Self:
        # Normalize to list of (key, value)
        items: list[tuple[int | str, AgentSet]]
        if isinstance(mapping, dict):
            items = list(mapping.items())
        else:
            items = list(mapping)

        obj = self._get_obj(inplace)

        # Helpers (build name->idx map only if needed)
        has_str_keys = any(isinstance(k, str) for k, _ in items)
        if has_str_keys:
            name_to_idx = {
                s.name: i for i, s in enumerate(obj._agentsets) if s.name is not None
            }

            def _find_index_by_name(name: str) -> int:
                try:
                    return name_to_idx[name]
                except KeyError:
                    raise KeyError(f"Agent set '{name}' not found")
        else:

            def _find_index_by_name(name: str) -> int:
                for i, s in enumerate(obj._agentsets):
                    if s.name == name:
                        return i

                raise KeyError(f"Agent set '{name}' not found")

        if atomic:
            n = len(obj._agentsets)
            # Map existing object identity -> index (for aliasing checks)
            id_to_idx = {id(s): i for i, s in enumerate(obj._agentsets)}

            for k, v in items:
                if not isinstance(v, AgentSet):
                    raise TypeError("Values must be AgentSet instances")
                if v.model is not obj.model:
                    raise TypeError(
                        "All AgentSets must belong to the same model as the registry"
                    )

                v_idx_existing = id_to_idx.get(id(v))

                if isinstance(k, int):
                    if not (0 <= k < n):
                        raise IndexError(
                            f"Index {k} out of range for AgentSetRegistry of size {n}"
                        )

                    # Prevent aliasing: the same object cannot appear in two positions
                    if v_idx_existing is not None and v_idx_existing != k:
                        raise ValueError(
                            f"This AgentSet instance already exists at index {v_idx_existing}; cannot also place it at {k}."
                        )

                    # Preserve name uniqueness when assigning by index
                    vname = v.name
                    if vname is not None:
                        try:
                            other_idx = _find_index_by_name(vname)
                            if other_idx != k:
                                raise ValueError(
                                    f"Duplicate agent set name disallowed: '{vname}' already at index {other_idx}"
                                )
                        except KeyError:
                            # name not present elsewhere -> OK
                            pass

                elif isinstance(k, str):
                    # Locate the slot by name; replacing that slot preserves uniqueness
                    idx = _find_index_by_name(k)

                    # Prevent aliasing: if the same object already exists at a different slot, forbid
                    if v_idx_existing is not None and v_idx_existing != idx:
                        raise ValueError(
                            f"This AgentSet instance already exists at index {v_idx_existing}; cannot also place it at {idx}."
                        )

                else:
                    raise TypeError("Keys must be int indices or str names")

        # Apply
        target = obj if inplace else obj.copy(deep=False)
        if not inplace:
            target._agentsets = list(obj._agentsets)

        for k, v in items:
            if isinstance(k, int):
                target._agentsets[k] = v  # keep v.name as-is (validated above)
            else:
                idx = _find_index_by_name(k)
                # Force the authoritative name without triggering external uniqueness checks
                if hasattr(v, "_name"):
                    v._name = k  # type: ignore[attr-defined]
                target._agentsets[idx] = v

        # Recompute ids cache
        target._recompute_ids()

        return target

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
=======
    @overload
    def contains(self, agents: int | AgentSet) -> bool: ...

    @overload
    def contains(self, agents: IdsLike | Iterable[AgentSet]) -> pl.Series: ...

    def contains(
        self, agents: IdsLike | AgentSet | Iterable[AgentSet]
    ) -> bool | pl.Series:
        if isinstance(agents, int):
            return agents in self._ids
        elif isinstance(agents, AgentSet):
            return self._check_agentsets_presence([agents]).any()
        elif isinstance(agents, Iterable):
            if len(agents) == 0:
                return True
            elif isinstance(next(iter(agents)), AgentSet):
                agents = cast(Iterable[AgentSet], agents)
                return self._check_agentsets_presence(list(agents))
            else:  # IdsLike
                agents = cast(IdsLike, agents)

                return pl.Series(agents, dtype=pl.UInt64).is_in(self._ids)
>>>>>>> 51c54cd666d876a5debb1b7dd71556ee9c458956

    @overload
    def do(
        self,
        method_name: str,
<<<<<<< HEAD
        *args: Any,
        sets: AgentSetSelector | None = None,
        return_results: Literal[False] = False,
        inplace: bool = True,
        key_by: KeyBy = "name",
        **kwargs: Any,
=======
        *args,
        mask: AgnosticAgentMask | IdsLike | dict[AgentSet, AgentMask] = None,
        return_results: Literal[False] = False,
        inplace: bool = True,
        **kwargs,
>>>>>>> 51c54cd666d876a5debb1b7dd71556ee9c458956
    ) -> Self: ...

    @overload
    def do(
        self,
        method_name: str,
<<<<<<< HEAD
        *args: Any,
        sets: AgentSetSelector,
        return_results: Literal[True],
        inplace: bool = True,
        key_by: KeyBy = "name",
        **kwargs: Any,
    ) -> dict[str, Any] | dict[int, Any] | dict[type[AgentSet], Any]: ...
=======
        *args,
        mask: AgnosticAgentMask | IdsLike | dict[AgentSet, AgentMask] = None,
        return_results: Literal[True],
        inplace: bool = True,
        **kwargs,
    ) -> dict[AgentSet, Any]: ...
>>>>>>> 51c54cd666d876a5debb1b7dd71556ee9c458956

    def do(
        self,
        method_name: str,
<<<<<<< HEAD
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
        # Normalize to a list of AgentSet instances using _resolve_selector
        selected = obj._resolve_selector(sets)  # type: ignore[arg-type]
        # Remove in reverse positional order
        indices = [i for i, s in enumerate(obj._agentsets) if s in selected]
        indices.sort(reverse=True)
        for idx in indices:
            obj._agentsets.pop(idx)
        # Recompute ids cache
        obj._recompute_ids()
        return obj

    def shuffle(self, inplace: bool = False) -> Self:
=======
        *args,
        mask: AgnosticAgentMask | IdsLike | dict[AgentSet, AgentMask] = None,
        return_results: bool = False,
        inplace: bool = True,
        **kwargs,
    ) -> Self | Any:
        obj = self._get_obj(inplace)
        agentsets_masks = obj._get_bool_masks(mask)
        if return_results:
            return {
                agentset: agentset.do(
                    method_name,
                    *args,
                    mask=mask,
                    return_results=return_results,
                    **kwargs,
                    inplace=inplace,
                )
                for agentset, mask in agentsets_masks.items()
            }
        else:
            obj._agentsets = [
                agentset.do(
                    method_name,
                    *args,
                    mask=mask,
                    return_results=return_results,
                    **kwargs,
                    inplace=inplace,
                )
                for agentset, mask in agentsets_masks.items()
            ]
            return obj

    def get(
        self,
        attr_names: str | Collection[str] | None = None,
        mask: AgnosticAgentMask | IdsLike | dict[AgentSet, AgentMask] = None,
    ) -> dict[AgentSet, Series] | dict[AgentSet, DataFrame]:
        agentsets_masks = self._get_bool_masks(mask)
        result = {}

        # Convert attr_names to list for consistent checking
        if attr_names is None:
            # None means get all data - no column filtering needed
            required_columns = []
        elif isinstance(attr_names, str):
            required_columns = [attr_names]
        else:
            required_columns = list(attr_names)

        for agentset, mask in agentsets_masks.items():
            # Fast column existence check - no data processing, just property access
            agentset_columns = agentset.df.columns

            # Check if all required columns exist in this agent set
            if not required_columns or all(
                col in agentset_columns for col in required_columns
            ):
                result[agentset] = agentset.get(attr_names, mask)

        return result

    def remove(
        self,
        agents: AgentSet | Iterable[AgentSet] | IdsLike,
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
>>>>>>> 51c54cd666d876a5debb1b7dd71556ee9c458956
        obj = self._get_obj(inplace)
        obj._agentsets = [agentset.shuffle(inplace=True) for agentset in obj._agentsets]
        return obj

    def sort(
        self,
        by: str | Sequence[str],
        ascending: bool | Sequence[bool] = True,
        inplace: bool = True,
<<<<<<< HEAD
        **kwargs: Any,
=======
        **kwargs,
>>>>>>> 51c54cd666d876a5debb1b7dd71556ee9c458956
    ) -> Self:
        obj = self._get_obj(inplace)
        obj._agentsets = [
            agentset.sort(by=by, ascending=ascending, inplace=inplace, **kwargs)
            for agentset in obj._agentsets
        ]
        return obj

<<<<<<< HEAD
=======
    def step(self, inplace: bool = True) -> Self:
        """Advance the state of the agents in the AgentSetRegistry by one step.

        Parameters
        ----------
        inplace : bool, optional
            Whether to update the AgentSetRegistry in place, by default True

        Returns
        -------
        Self
        """
        obj = self._get_obj(inplace)
        for agentset in obj._agentsets:
            agentset.step()
        return obj

>>>>>>> 51c54cd666d876a5debb1b7dd71556ee9c458956
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

<<<<<<< HEAD
    def _recompute_ids(self) -> None:
        """Rebuild the registry-level `unique_id` cache from current AgentSets.

        Ensures `self._ids` stays a `pl.UInt64` Series and empty when no sets.
        """
        if self._agentsets:
            cols = [pl.Series(s["unique_id"]) for s in self._agentsets]
            self._ids = (
                pl.concat(cols)
                if cols
                else pl.Series(name="unique_id", dtype=pl.UInt64)
            )
        else:
            self._ids = pl.Series(name="unique_id", dtype=pl.UInt64)

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
=======
    def _get_bool_masks(
        self,
        mask: (AgnosticAgentMask | IdsLike | dict[AgentSet, AgentMask]) = None,
    ) -> dict[AgentSet, BoolSeries]:
        return_dictionary = {}
        if not isinstance(mask, dict):
            # No need to convert numpy integers - let polars handle them directly
            mask = {agentset: mask for agentset in self._agentsets}
        for agentset, mask_value in mask.items():
            return_dictionary[agentset] = agentset._get_bool_mask(mask_value)
        return return_dictionary
>>>>>>> 51c54cd666d876a5debb1b7dd71556ee9c458956

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

<<<<<<< HEAD
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
=======
    def __add__(self, other: AgentSet | Iterable[AgentSet]) -> Self:
        """Add AgentSets to a new AgentSetRegistry through the + operator.

        Parameters
        ----------
        other : AgentSet | Iterable[AgentSet]
            The AgentSets to add.

        Returns
        -------
        Self
            A new AgentSetRegistry with the added AgentSets.
        """
        return super().__add__(other)

    def __getattr__(self, name: str) -> dict[AgentSet, Any]:
        # Avoids infinite recursion of private attributes
        if __debug__:  # Only execute in non-optimized mode
            if name.startswith("_"):
                raise AttributeError(
                    f"'{self.__class__.__name__}' object has no attribute '{name}'"
                )
        return {agentset: getattr(agentset, name) for agentset in self._agentsets}

    @overload
    def __getitem__(
        self, key: str | tuple[dict[AgentSet, AgentMask], str]
    ) -> dict[AgentSet, Series | pl.Expr]: ...

    @overload
    def __getitem__(
        self,
        key: (
            Collection[str]
            | AgnosticAgentMask
            | IdsLike
            | tuple[dict[AgentSet, AgentMask], Collection[str]]
        ),
    ) -> dict[AgentSet, DataFrame]: ...

    def __getitem__(
        self,
        key: (
            str
            | Collection[str]
            | AgnosticAgentMask
            | IdsLike
            | tuple[dict[AgentSet, AgentMask], str]
            | tuple[dict[AgentSet, AgentMask], Collection[str]]
        ),
    ) -> dict[AgentSet, Series | pl.Expr] | dict[AgentSet, DataFrame]:
        return super().__getitem__(key)

    def __iadd__(self, agents: AgentSet | Iterable[AgentSet]) -> Self:
        """Add AgentSets to the AgentSetRegistry through the += operator.

        Parameters
        ----------
        agents : AgentSet | Iterable[AgentSet]
            The AgentSets to add.

        Returns
        -------
        Self
            The updated AgentSetRegistry.
        """
        return super().__iadd__(agents)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return (agent for agentset in self._agentsets for agent in iter(agentset))

    def __isub__(self, agents: AgentSet | Iterable[AgentSet] | IdsLike) -> Self:
        """Remove AgentSets from the AgentSetRegistry through the -= operator.

        Parameters
        ----------
        agents : AgentSet | Iterable[AgentSet] | IdsLike
            The AgentSets or agent IDs to remove.

        Returns
        -------
        Self
            The updated AgentSetRegistry.
        """
        return super().__isub__(agents)

    def __len__(self) -> int:
        return sum(len(agentset._df) for agentset in self._agentsets)
>>>>>>> 51c54cd666d876a5debb1b7dd71556ee9c458956

    def __repr__(self) -> str:
        return "\n".join([repr(agentset) for agentset in self._agentsets])

<<<<<<< HEAD
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
        self._recompute_ids()
=======
    def __reversed__(self) -> Iterator:
        return (
            agent
            for agentset in self._agentsets
            for agent in reversed(agentset._backend)
        )

    def __setitem__(
        self,
        key: (
            str
            | Collection[str]
            | AgnosticAgentMask
            | IdsLike
            | tuple[dict[AgentSet, AgentMask], str]
            | tuple[dict[AgentSet, AgentMask], Collection[str]]
        ),
        values: Any,
    ) -> None:
        super().__setitem__(key, values)
>>>>>>> 51c54cd666d876a5debb1b7dd71556ee9c458956

    def __str__(self) -> str:
        return "\n".join([str(agentset) for agentset in self._agentsets])

<<<<<<< HEAD
    def keys(self, *, key_by: KeyBy = "name") -> Iterable[Any]:
        if key_by not in ("name", "index", "type"):
            raise ValueError("key_by must be 'name'|'index'|'type'")
        if key_by == "index":
            yield from range(len(self._agentsets))
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

    @property
    def ids(self) -> pl.Series:
        """Public view of all agent unique_id values across contained sets."""
        return self._ids

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
=======
    def __sub__(self, agents: AgentSet | Iterable[AgentSet] | IdsLike) -> Self:
        """Remove AgentSets from a new AgentSetRegistry through the - operator.

        Parameters
        ----------
        agents : AgentSet | Iterable[AgentSet] | IdsLike
            The AgentSets or agent IDs to remove. Supports NumPy integer types.

        Returns
        -------
        Self
            A new AgentSetRegistry with the removed AgentSets.
        """
        return super().__sub__(agents)

    @property
    def df(self) -> dict[AgentSet, DataFrame]:
        return {agentset: agentset.df for agentset in self._agentsets}

    @df.setter
    def df(self, other: Iterable[AgentSet]) -> None:
        """Set the agents in the AgentSetRegistry.

        Parameters
        ----------
        other : Iterable[AgentSet]
            The AgentSets to set.
        """
        self._agentsets = list(other)

    @property
    def active_agents(self) -> dict[AgentSet, DataFrame]:
        return {agentset: agentset.active_agents for agentset in self._agentsets}

    @active_agents.setter
    def active_agents(
        self, agents: AgnosticAgentMask | IdsLike | dict[AgentSet, AgentMask]
    ) -> None:
        self.select(agents, inplace=True)

    @property
    def agentsets_by_type(self) -> dict[type[AgentSet], Self]:
        """Get the agent sets in the AgentSetRegistry grouped by type.

        Returns
        -------
        dict[type[AgentSet], Self]
            A dictionary mapping agent set types to the corresponding AgentSetRegistry.
        """

        def copy_without_agentsets() -> Self:
            return self.copy(deep=False, skip=["_agentsets"])

        dictionary = defaultdict(copy_without_agentsets)

        for agentset in self._agentsets:
            agents_df = dictionary[agentset.__class__]
            agents_df._agentsets = []
            agents_df._agentsets = agents_df._agentsets + [agentset]
            dictionary[agentset.__class__] = agents_df
        return dictionary

    @property
    def inactive_agents(self) -> dict[AgentSet, DataFrame]:
        return {agentset: agentset.inactive_agents for agentset in self._agentsets}

    @property
    def index(self) -> dict[AgentSet, Index]:
        return {agentset: agentset.index for agentset in self._agentsets}

    @property
    def pos(self) -> dict[AgentSet, DataFrame]:
        return {agentset: agentset.pos for agentset in self._agentsets}
>>>>>>> 51c54cd666d876a5debb1b7dd71556ee9c458956
