"""
Concrete implementation of the agents collection for mesa-frames.

This module provides the concrete implementation of the agents collection class
for the mesa-frames library. It defines the AgentsDF class, which serves as a
container for all agent sets in a model, leveraging DataFrame-based storage for
improved performance.

Classes:
    AgentsDF(AgentContainer):
        A collection of AgentSetDFs. This class acts as a container for all
        agents in the model, organizing them into separate AgentSetDF instances
        based on their types.

The AgentsDF class is designed to be used within ModelDF instances to manage
all agents in the simulation. It provides methods for adding, removing, and
accessing agents and agent sets, while taking advantage of the performance
benefits of DataFrame-based agent storage.

Usage:
    The AgentsDF class is typically instantiated and used within a ModelDF subclass:

    from mesa_frames.concrete.model import ModelDF
    from mesa_frames.concrete.agents import AgentsDF
    from mesa_frames.concrete import AgentSetPolars

    class MyCustomModel(ModelDF):
        def __init__(self):
            super().__init__()
            # Adding agent sets to the collection
            self.agents += AgentSetPolars(self)
            self.agents += AnotherAgentSetPolars(self)

        def step(self):
            # Step all agent sets
            self.agents.do("step")

Note:
    This concrete implementation builds upon the abstract AgentContainer class
    defined in the mesa_frames.abstract package, providing a ready-to-use
    agents collection that integrates with the DataFrame-based agent storage system.

For more detailed information on the AgentsDF class and its methods, refer to
the class docstring.
"""

from __future__ import annotations  # For forward references

from collections.abc import Callable, Collection, Iterable, Iterator, Sequence
from typing import Any, Literal, Self, cast, overload

import numpy as np
import polars as pl

from mesa_frames.abstract.agents import AgentContainer, AgentSetDF
from mesa_frames.concrete.accessors import AgentSetsAccessor
from mesa_frames.types_ import (
    AgentMask,
    AgnosticAgentMask,
    BoolSeries,
    DataFrame,
    IdsLike,
    Index,
    KeyBy,
    Series,
)


class AgentsDF(AgentContainer):
    """A collection of AgentSetDFs. All agents of the model are stored here."""

    # Do not copy the accessor; it holds a reference to this instance and is
    # cheaply re-created on demand via the `sets` property.
    _skip_copy: list[str] = ["_sets_accessor"]
    _agentsets: list[AgentSetDF]
    _ids: pl.Series

    def __init__(self, model: mesa_frames.concrete.model.ModelDF) -> None:
        """Initialize a new AgentsDF.

        Parameters
        ----------
        model : mesa_frames.concrete.model.ModelDF
            The model associated with the AgentsDF.
        """
        self._model = model
        self._agentsets = []  # internal storage; used by AgentSetsAccessor
        self._ids = pl.Series(name="unique_id", dtype=pl.UInt64)
        # Accessor is created lazily in the property to survive copy/deepcopy
        self._sets_accessor = AgentSetsAccessor(self)

    @property
    def sets(self) -> AgentSetsAccessor:
        """Accessor for agentset lookup by index/name/type.

        Does not conflict with AgentsDF's existing __getitem__ column API.
        """
        # Ensure accessor always points to this instance (robust to copy/deepcopy)
        acc = getattr(self, "_sets_accessor", None)
        if acc is None or getattr(acc, "_parent", None) is not self:
            acc = AgentSetsAccessor(self)
            self._sets_accessor = acc
        return acc

    @staticmethod
    def _make_unique_name(base: str, existing: set[str]) -> str:
        """Generate a unique name by appending numeric suffix if needed."""
        if base not in existing:
            return base
        # If ends with _<int>, increment; else append _1
        import re

        m = re.match(r"^(.*?)(?:_(\d+))$", base)
        if m:
            prefix, num = m.group(1), int(m.group(2))
            nxt = num + 1
            candidate = f"{prefix}_{nxt}"
            while candidate in existing:
                nxt += 1
                candidate = f"{prefix}_{nxt}"
            return candidate
        else:
            candidate = f"{base}_1"
            i = 1
            while candidate in existing:
                i += 1
                candidate = f"{base}_{i}"
            return candidate

    def _canonicalize_names(self, new_agentsets: list[AgentSetDF]) -> None:
        """Canonicalize names across existing + new agent sets, ensuring uniqueness."""
        existing_names = {s.name for s in self._agentsets}

        # Process each new agent set in batch to handle potential conflicts
        for aset in new_agentsets:
            # Use the static method to generate unique name
            unique_name = self._make_unique_name(aset.name, existing_names)
            if unique_name != aset.name:
                # Directly set the name instead of calling rename
                import warnings

                warnings.warn(
                    f"AgentSet with name '{aset.name}' already exists; renamed to '{unique_name}'.",
                    UserWarning,
                    stacklevel=2,
                )
            aset._name = unique_name
            existing_names.add(unique_name)

    def _rename_sets(
        self,
        target: AgentSetDF
        | str
        | dict[AgentSetDF | str, str]
        | list[tuple[AgentSetDF | str, str]],
        new_name: str | None = None,
        *,
        on_conflict: Literal["canonicalize", "raise"] = "canonicalize",
        mode: Literal["atomic", "best_effort"] = "atomic",
    ) -> str | dict[AgentSetDF, str]:
        """Handle agent set renaming delegations from accessor.

        Parameters
        ----------
        target : AgentSetDF | str | dict[AgentSetDF | str, str] | list[tuple[AgentSetDF | str, str]]
            Either:
            - Single: AgentSet or name string (must provide new_name)
            - Batch: {target: new_name} dict or [(target, new_name), ...] list
        new_name : str | None, optional
            New name (only used for single renames)
        on_conflict : Literal["canonicalize", "raise"]
            Conflict resolution: "canonicalize" (default) appends suffixes, "raise" raises ValueError
        mode : Literal["atomic", "best_effort"]
            Rename mode: "atomic" applies all or none (default), "best_effort" skips failed renames

        Returns
        -------
        str | dict[AgentSetDF, str]
            Single rename: final name string
            Batch: {agentset: final_name} mapping

        Raises
        ------
        ValueError
            If target format is invalid or single rename missing new_name
        KeyError
            If agent set name not found or naming conflicts with raise mode
        """
        # Parse different target formats and build rename operations
        rename_ops = self._parse_rename_target(target, new_name)

        # Map on_conflict values to _rename_single_set expected values
        mapped_on_conflict = "error" if on_conflict == "raise" else "overwrite"

        # Determine if this is single or batch based on the input format
        if isinstance(target, (str, AgentSetDF)):
            # Single rename - return the final name
            target_set, new_name = rename_ops[0]
            return self._rename_single_set(
                target_set, new_name, on_conflict=mapped_on_conflict, mode="atomic"
            )
        else:
            # Batch rename (dict or list) - return mapping of original sets to final names
            result = {}
            for target_set, new_name in rename_ops:
                final_name = self._rename_single_set(
                    target_set, new_name, on_conflict=mapped_on_conflict, mode="atomic"
                )
                result[target_set] = final_name
            return result

    def _parse_rename_target(
        self,
        target: AgentSetDF
        | str
        | dict[AgentSetDF | str, str]
        | list[tuple[AgentSetDF | str, str]],
        new_name: str | None = None,
    ) -> list[tuple[AgentSetDF, str]]:
        """Parse the target parameter into a list of (agentset, new_name) pairs."""
        rename_ops = []
        # Get available names for error messages
        available_names = [getattr(s, "name", None) for s in self._agentsets]

        if isinstance(target, dict):
            # target is a dict mapping agent sets/names to new names
            for k, v in target.items():
                if isinstance(k, str):
                    # k is a name, find the agent set
                    target_set = None
                    for aset in self._agentsets:
                        if aset.name == k:
                            target_set = aset
                            break
                    if target_set is None:
                        raise KeyError(
                            f"No agent set named '{k}'. Available: {available_names}"
                        )
                else:
                    # k is an AgentSetDF
                    target_set = k
                rename_ops.append((target_set, v))

        elif isinstance(target, list):
            # target is a list of (agent_set/name, new_name) tuples
            for k, v in target:
                if isinstance(k, str):
                    # k is a name, find the agent set
                    target_set = None
                    for aset in self._agentsets:
                        if aset.name == k:
                            target_set = aset
                            break
                    if target_set is None:
                        raise KeyError(
                            f"No agent set named '{k}'. Available: {available_names}"
                        )
                else:
                    # k is an AgentSetDF
                    target_set = k
                rename_ops.append((target_set, v))

        else:
            # target is single AgentSetDF or name, new_name must be provided
            if isinstance(target, str):
                # target is a name, find the agent set
                target_set = None
                for aset in self._agentsets:
                    if aset.name == target:
                        target_set = aset
                        break
                if target_set is None:
                    raise KeyError(
                        f"No agent set named '{target}'. Available: {available_names}"
                    )
            else:
                # target is an AgentSetDF
                target_set = target

            if new_name is None:
                raise ValueError("new_name must be provided for single rename")
            rename_ops.append((target_set, new_name))

        return rename_ops

    def _rename_single_set(
        self,
        target: AgentSetDF,
        new_name: str,
        on_conflict: Literal["error", "skip", "overwrite"] = "error",
        mode: Literal["atomic"] = "atomic",
    ) -> str:
        """Handle single agent set renaming.

        Parameters
        ----------
        target : AgentSetDF
            The agent set to rename
        new_name : str
            The new name for the agent set
        on_conflict : Literal["error", "skip", "overwrite"], optional
            How to handle naming conflicts, by default 'error'
        mode : Literal["atomic"], optional
            Rename mode, by default 'atomic'

        Returns
        -------
        str
            The final name assigned to the agent set

        Raises
        ------
        ValueError
            If target is not in this container or other validation errors
        KeyError
            If on_conflict='error' and new_name conflicts with existing set
        """
        # Validate target is in this container
        if target not in self._agentsets:
            available_names = [s.name for s in self._agentsets]
            raise ValueError(
                f"AgentSet {target} is not in this container. "
                f"Available agent sets: {available_names}"
            )

        # Check for conflicts with existing names (excluding current target)
        existing_names = {s.name for s in self._agentsets if s is not target}
        if new_name in existing_names:
            if on_conflict == "error":
                available_names = [
                    s.name for s in self._agentsets if s.name != target.name
                ]
                raise KeyError(
                    f"AgentSet name '{new_name}' already exists. Available names: {available_names}"
                )
            elif on_conflict == "skip":
                # Return existing name without changes
                return target._name
            # on_conflict == 'overwrite' - proceed with rename

        # Apply name canonicalization if needed
        final_name = self._make_unique_name(new_name, existing_names)
        target._name = final_name
        return final_name

    def add(
        self,
        agents: AgentSetDF | Iterable[AgentSetDF],
        inplace: bool = True,
    ) -> Self:
        """Add an AgentSetDF to the AgentsDF (only gate for name validation).

        Parameters
        ----------
        agents : AgentSetDF | Iterable[AgentSetDF]
            The AgentSetDFs to add.
        inplace : bool, optional
            Whether to add the AgentSetDFs in place. Defaults to True.

        Returns
        -------
        Self
            The updated AgentsDF.

        Raises
        ------
        ValueError
            If any AgentSetDFs are already present or if IDs are not unique.
        """
        obj = self._get_obj(inplace)
        other_list = obj._return_agentsets_list(agents)
        if obj._check_agentsets_presence(other_list).any():
            raise ValueError("Some agentsets are already present in the AgentsDF.")

        # Validate and canonicalize names across existing + batch before mutating
        obj._canonicalize_names(other_list)

        # Collect unique_ids from agent sets that have them (may be empty at this point)
        new_ids_list = [obj._ids]
        for agentset in other_list:
            if len(agentset) > 0:  # Only include if there are agents in the set
                new_ids_list.append(agentset["unique_id"])

        new_ids = pl.concat(new_ids_list)
        if new_ids.is_duplicated().any():
            raise ValueError("Some of the agent IDs are not unique.")

        obj._agentsets.extend(other_list)
        obj._ids = new_ids

        return obj

    @overload
    def contains(self, agents: int | AgentSetDF) -> bool: ...

    @overload
    def contains(self, agents: IdsLike | Iterable[AgentSetDF]) -> pl.Series: ...

    def contains(
        self, agents: IdsLike | AgentSetDF | Iterable[AgentSetDF]
    ) -> bool | pl.Series:
        if isinstance(agents, int):
            return agents in self._ids
        elif isinstance(agents, AgentSetDF):
            return self._check_agentsets_presence([agents]).any()
        elif isinstance(agents, Iterable):
            if len(agents) == 0:
                return True
            elif isinstance(next(iter(agents)), AgentSetDF):
                agents = cast(Iterable[AgentSetDF], agents)
                return self._check_agentsets_presence(list(agents))
            else:  # IdsLike
                agents = cast(IdsLike, agents)

                return pl.Series(agents, dtype=pl.UInt64).is_in(self._ids)

    @overload
    def do(
        self,
        method_name: str,
        *args,
        mask: AgnosticAgentMask | IdsLike | dict[AgentSetDF, AgentMask] = None,
        return_results: Literal[False] = False,
        inplace: bool = True,
        **kwargs,
    ) -> Self: ...

    @overload
    def do(
        self,
        method_name: str,
        *args,
        mask: AgnosticAgentMask | IdsLike | dict[AgentSetDF, AgentMask] = None,
        return_results: Literal[True],
        inplace: bool = True,
        **kwargs,
    ) -> dict[AgentSetDF, Any]: ...

    def do(
        self,
        method_name: str,
        *args,
        mask: AgnosticAgentMask | IdsLike | dict[AgentSetDF, AgentMask] = None,
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
        mask: AgnosticAgentMask | IdsLike | dict[AgentSetDF, AgentMask] = None,
        key_by: KeyBy = "name",
    ) -> (
        dict[AgentSetDF, Series]
        | dict[AgentSetDF, DataFrame]
        | dict[str, Any]
        | dict[int, Any]
        | dict[type, Any]
    ):
        agentsets_masks = self._get_bool_masks(mask)
        result: dict[AgentSetDF, Any] = {}

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

        if key_by == "name":
            return {cast(AgentSetDF, a).name: v for a, v in result.items()}  # type: ignore[return-value]
        elif key_by == "index":
            index_map = {agentset: i for i, agentset in enumerate(self._agentsets)}
            return {index_map[a]: v for a, v in result.items()}  # type: ignore[return-value]
        elif key_by == "type":
            return {type(a): v for a, v in result.items()}  # type: ignore[return-value]
        else:
            raise ValueError("key_by must be one of 'name', 'index', or 'type'")

    def remove(
        self,
        agents: AgentSetDF | Iterable[AgentSetDF] | IdsLike,
        inplace: bool = True,
    ) -> Self:
        obj = self._get_obj(inplace)
        if agents is None or (isinstance(agents, Iterable) and len(agents) == 0):
            return obj
        if isinstance(agents, AgentSetDF):
            agents = [agents]
        if isinstance(agents, Iterable) and isinstance(next(iter(agents)), AgentSetDF):
            # We have to get the index of the original AgentSetDF because the copy made AgentSetDFs with different hash
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
        mask: AgnosticAgentMask | IdsLike | dict[AgentSetDF, AgentMask] = None,
        filter_func: Callable[[AgentSetDF], AgentMask] | None = None,
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
        attr_names: str | dict[AgentSetDF, Any] | Collection[str],
        values: Any | None = None,
        mask: AgnosticAgentMask | IdsLike | dict[AgentSetDF, AgentMask] = None,
        inplace: bool = True,
    ) -> Self:
        obj = self._get_obj(inplace)
        agentsets_masks = obj._get_bool_masks(mask)
        if isinstance(attr_names, dict):
            for agentset, values in attr_names.items():
                if not inplace:
                    # We have to get the index of the original AgentSetDF because the copy made AgentSetDFs with different hash
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
        obj = self._get_obj(inplace)
        obj._agentsets = [agentset.shuffle(inplace=True) for agentset in obj._agentsets]
        return obj

    def sort(
        self,
        by: str | Sequence[str],
        ascending: bool | Sequence[bool] = True,
        inplace: bool = True,
        **kwargs,
    ) -> Self:
        obj = self._get_obj(inplace)
        obj._agentsets = [
            agentset.sort(by=by, ascending=ascending, inplace=inplace, **kwargs)
            for agentset in obj._agentsets
        ]
        return obj

    def step(self, inplace: bool = True) -> Self:
        """Advance the state of the agents in the AgentsDF by one step.

        Parameters
        ----------
        inplace : bool, optional
            Whether to update the AgentsDF in place, by default True

        Returns
        -------
        Self
        """
        obj = self._get_obj(inplace)
        for agentset in obj._agentsets:
            agentset.step()
        return obj

    def _check_ids_presence(self, other: list[AgentSetDF]) -> pl.DataFrame:
        """Check if the IDs of the agents to be added are unique.

        Parameters
        ----------
        other : list[AgentSetDF]
            The AgentSetDFs to check.

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

    def _check_agentsets_presence(self, other: list[AgentSetDF]) -> pl.Series:
        """Check if the agent sets to be added are already present in the AgentsDF.

        Parameters
        ----------
        other : list[AgentSetDF]
            The AgentSetDFs to check.

        Returns
        -------
        pl.Series
            A boolean Series indicating if the agent sets are present.

        Raises
        ------
        ValueError
            If the agent sets are already present in the AgentsDF.
        """
        other_set = set(other)
        return pl.Series(
            [agentset in other_set for agentset in self._agentsets], dtype=pl.Boolean
        )

    def _get_bool_masks(
        self,
        mask: (AgnosticAgentMask | IdsLike | dict[AgentSetDF, AgentMask]) = None,
    ) -> dict[AgentSetDF, BoolSeries]:
        return_dictionary = {}
        if not isinstance(mask, dict):
            # No need to convert numpy integers - let polars handle them directly
            mask = {agentset: mask for agentset in self._agentsets}
        for agentset, mask_value in mask.items():
            return_dictionary[agentset] = agentset._get_bool_mask(mask_value)
        return return_dictionary

    def _return_agentsets_list(
        self, agentsets: AgentSetDF | Iterable[AgentSetDF]
    ) -> list[AgentSetDF]:
        """Convert the agentsets to a list of AgentSetDF.

        Parameters
        ----------
        agentsets : AgentSetDF | Iterable[AgentSetDF]

        Returns
        -------
        list[AgentSetDF]
        """
        return [agentsets] if isinstance(agentsets, AgentSetDF) else list(agentsets)

    def __add__(self, other: AgentSetDF | Iterable[AgentSetDF]) -> Self:
        """Add AgentSetDFs to a new AgentsDF through the + operator.

        Parameters
        ----------
        other : AgentSetDF | Iterable[AgentSetDF]
            The AgentSetDFs to add.

        Returns
        -------
        Self
            A new AgentsDF with the added AgentSetDFs.
        """
        return super().__add__(other)

    def __getattr__(self, name: str) -> dict[str, Any]:
        # Avoids infinite recursion of private attributes
        if __debug__:  # Only execute in non-optimized mode
            if name.startswith("_"):
                raise AttributeError(
                    f"'{self.__class__.__name__}' object has no attribute '{name}'"
                )
        return {agentset: getattr(agentset, name) for agentset in self._agentsets}

    @overload
    def __getitem__(
        self, key: str | tuple[dict[AgentSetDF, AgentMask], str]
    ) -> dict[AgentSetDF, Series | pl.Expr]: ...

    @overload
    def __getitem__(
        self,
        key: (
            Collection[str]
            | AgnosticAgentMask
            | IdsLike
            | tuple[dict[AgentSetDF, AgentMask], Collection[str]]
        ),
    ) -> dict[AgentSetDF, DataFrame]: ...

    def __getitem__(
        self,
        key: (
            str
            | Collection[str]
            | AgnosticAgentMask
            | IdsLike
            | tuple[dict[AgentSetDF, AgentMask], str]
            | tuple[dict[AgentSetDF, AgentMask], Collection[str]]
        ),
    ) -> dict[AgentSetDF, Series | pl.Expr] | dict[AgentSetDF, DataFrame]:
        return super().__getitem__(key)

    def __iadd__(self, agents: AgentSetDF | Iterable[AgentSetDF]) -> Self:
        """Add AgentSetDFs to the AgentsDF through the += operator.

        Parameters
        ----------
        agents : AgentSetDF | Iterable[AgentSetDF]
            The AgentSetDFs to add.

        Returns
        -------
        Self
            The updated AgentsDF.
        """
        return super().__iadd__(agents)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return (agent for agentset in self._agentsets for agent in iter(agentset))

    def __isub__(self, agents: AgentSetDF | Iterable[AgentSetDF] | IdsLike) -> Self:
        """Remove AgentSetDFs from the AgentsDF through the -= operator.

        Parameters
        ----------
        agents : AgentSetDF | Iterable[AgentSetDF] | IdsLike
            The AgentSetDFs or agent IDs to remove.

        Returns
        -------
        Self
            The updated AgentsDF.
        """
        return super().__isub__(agents)

    def __len__(self) -> int:
        return sum(len(agentset._df) for agentset in self._agentsets)

    def __repr__(self) -> str:
        return "\n".join([repr(agentset) for agentset in self._agentsets])

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
            | tuple[dict[AgentSetDF, AgentMask], str]
            | tuple[dict[AgentSetDF, AgentMask], Collection[str]]
        ),
        values: Any,
    ) -> None:
        super().__setitem__(key, values)

    def __str__(self) -> str:
        return "\n".join([str(agentset) for agentset in self._agentsets])

    def __sub__(self, agents: AgentSetDF | Iterable[AgentSetDF] | IdsLike) -> Self:
        """Remove AgentSetDFs from a new AgentsDF through the - operator.

        Parameters
        ----------
        agents : AgentSetDF | Iterable[AgentSetDF] | IdsLike
            The AgentSetDFs or agent IDs to remove. Supports NumPy integer types.

        Returns
        -------
        Self
            A new AgentsDF with the removed AgentSetDFs.
        """
        return super().__sub__(agents)

    @property
    def df(self) -> dict[AgentSetDF, DataFrame]:
        return {agentset: agentset.df for agentset in self._agentsets}

    @df.setter
    def df(self, other: Iterable[AgentSetDF]) -> None:
        """Set the agents in the AgentsDF.

        Parameters
        ----------
        other : Iterable[AgentSetDF]
            The AgentSetDFs to set.
        """
        self._agentsets = list(other)

    @property
    def active_agents(self) -> dict[AgentSetDF, DataFrame]:
        return {agentset: agentset.active_agents for agentset in self._agentsets}

    @active_agents.setter
    def active_agents(
        self, agents: AgnosticAgentMask | IdsLike | dict[AgentSetDF, AgentMask]
    ) -> None:
        self.select(agents, inplace=True)

    @property
    def inactive_agents(self) -> dict[AgentSetDF, DataFrame]:
        return {agentset: agentset.inactive_agents for agentset in self._agentsets}

    @property
    def index(self) -> dict[AgentSetDF, Index]:
        return {agentset: agentset.index for agentset in self._agentsets}

    @property
    def pos(self) -> dict[AgentSetDF, DataFrame]:
        return {agentset: agentset.pos for agentset in self._agentsets}
