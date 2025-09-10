"""
Polars-based implementation of AgentSet for mesa-frames.

This module provides a concrete implementation of the AgentSet class using Polars
as the backend for DataFrame operations. It defines the AgentSet class,
which combines the abstract AbstractAgentSet functionality with Polars-specific
operations for efficient agent management and manipulation.

Classes:
    AgentSet(AbstractAgentSet, PolarsMixin):
        A Polars-based implementation of the AgentSet. This class uses Polars
        DataFrames to store and manipulate agent data, providing high-performance
        operations for large numbers of agents.

The AgentSet class is designed to be used within Model instances or as
part of an AgentSetRegistry collection. It leverages the power of Polars for fast and
efficient data operations on agent attributes and behaviors.

Usage:
    The AgentSet class can be used directly in a model or as part of an
    AgentSetRegistry collection:

    from mesa_frames.concrete.model import Model
    from mesa_frames.concrete.agentset import AgentSet
    import polars as pl

    class MyAgents(AgentSet):
        def __init__(self, model):
            super().__init__(model)
            # Initialize with some agents
            self.add(pl.DataFrame({'id': range(100), 'wealth': 10}))

        def step(self):
            # Implement step behavior using Polars operations
            self.sets = self.sets.with_columns(new_wealth = pl.col('wealth') + 1)

    class MyModel(Model):
        def __init__(self):
            super().__init__()
            self.sets += MyAgents(self)

        def step(self):
            self.sets.step()

Features:
    - Efficient storage and manipulation of large agent populations
    - Fast vectorized operations on agent attributes
    - Support for lazy evaluation and query optimization
    - Seamless integration with other mesa-frames components

Note:
    This implementation relies on Polars, so users should ensure that Polars
    is installed and imported. The performance characteristics of this class
    will depend on the Polars version and the specific operations used.

For more detailed information on the AgentSet class and its methods,
refer to the class docstring.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Iterator, Sequence
from typing import Any, Literal, Self, overload

import numpy as np
import polars as pl

from mesa_frames.abstract.agentset import AbstractAgentSet
from mesa_frames.concrete.mixin import PolarsMixin
from mesa_frames.types_ import AgentMask, AgentPolarsMask, IntoExpr, PolarsIdsLike
from mesa_frames.utils import copydoc


@copydoc(AbstractAgentSet)
class AgentSet(AbstractAgentSet, PolarsMixin):
    """Polars-based implementation of AgentSet."""

    _df: pl.DataFrame
    _copy_with_method: dict[str, tuple[str, list[str]]] = {
        "_df": ("clone", []),
    }
    _copy_only_reference: list[str] = ["_model", "_mask"]
    _mask: pl.Expr | pl.Series

    def __init__(self, model: Model, name: str | None = None) -> None:
        """Initialize a new AgentSet.

        Parameters
        ----------
        model : "mesa_frames.concrete.model.Model"
            The model that the agent set belongs to.
        name : str | None, optional
            Name for this agent set. If None, class name is used.
            Will be converted to snake_case if in camelCase.
        """
        # Model reference
        self._model = model
        # Set proposed name (no uniqueness guarantees here)
        self._name = name if name is not None else self.__class__.__name__
        # No definition of schema with unique_id, as it becomes hard to add new agents
        self._df = pl.DataFrame()
        self._mask = pl.repeat(True, len(self._df), dtype=pl.Boolean, eager=True)

    def rename(self, new_name: str) -> str:
        """Rename this agent set. If attached to AgentSetRegistry, delegate for uniqueness enforcement.

        Parameters
        ----------
        new_name : str
            Desired new name.

        Returns
        -------
        str
            The final name used (may be canonicalized if duplicates exist).

        Raises
        ------
        ValueError
            If name conflicts occur and delegate encounters errors.
        """
        # Always delegate to the container's accessor if available through the model's sets
        # Check if we have a model and can find the AgentSetRegistry that contains this set
        if self in self.model.sets:
            return self.model.sets.rename(self._name, new_name)

        # Set name locally if no container found
        self._name = new_name
        return new_name

    def add(
        self,
        agents: pl.DataFrame | Sequence[Any] | dict[str, Any],
        inplace: bool = True,
    ) -> Self:
        """Add agents to the AgentSet.

        Parameters
        ----------
        agents : pl.DataFrame | Sequence[Any] | dict[str, Any]
            The agents to add.
        inplace : bool, optional
            Whether to add the agents in place, by default True.

        Returns
        -------
        Self
            The updated AgentSet.
        """
        obj = self._get_obj(inplace)
        if isinstance(agents, AbstractAgentSet):
            raise TypeError(
                "AgentSet.add() does not accept AgentSet objects. "
                "Extract the DataFrame with agents.agents.drop('unique_id') first."
            )
        elif isinstance(agents, pl.DataFrame):
            if "unique_id" in agents.columns:
                raise ValueError("Dataframe should not have a unique_id column.")
            new_agents = agents
        elif isinstance(agents, dict):
            if "unique_id" in agents:
                raise ValueError("Dictionary should not have a unique_id key.")
            new_agents = pl.DataFrame(agents)
        else:  # Sequence
            if len(obj._df) != 0:
                # For non-empty AgentSet, check column count
                expected_columns = len(obj._df.columns) - 1  # Exclude unique_id
                if len(agents) != expected_columns:
                    raise ValueError(
                        f"Length of data ({len(agents)}) must match the number of columns in the AgentSet (excluding unique_id): {expected_columns}"
                    )
                new_agents = pl.DataFrame(
                    [list(agents)],
                    schema=[col for col in obj._df.schema if col != "unique_id"],
                    orient="row",
                )
            else:
                # For empty AgentSet, cannot infer schema from sequence
                raise ValueError(
                    "Cannot add a sequence to an empty AgentSet. Use a DataFrame or dict with column names."
                )

        new_agents = new_agents.with_columns(
            self._generate_unique_ids(len(new_agents)).alias("unique_id")
        )

        # If self._mask is pl.Expr, then new mask is the same.
        # If self._mask is pl.Series[bool], then new mask has to be updated.
        originally_empty = len(obj._df) == 0
        if isinstance(obj._mask, pl.Series) and not originally_empty:
            original_active_indices = obj._df.filter(obj._mask)["unique_id"]

        obj._df = pl.concat([obj._df, new_agents], how="diagonal_relaxed")

        if isinstance(obj._mask, pl.Series) and not originally_empty:
            obj._update_mask(original_active_indices, new_agents["unique_id"])

        return obj

    @overload
    def contains(self, agents: int) -> bool: ...

    @overload
    def contains(self, agents: PolarsIdsLike) -> pl.Series: ...

    def contains(
        self,
        agents: PolarsIdsLike,
    ) -> bool | pl.Series:
        if isinstance(agents, pl.Series):
            return agents.is_in(self._df["unique_id"])
        elif isinstance(agents, Collection) and not isinstance(agents, str):
            return pl.Series(agents, dtype=pl.UInt64).is_in(self._df["unique_id"])
        else:
            return agents in self._df["unique_id"]

    @overload
    def do(
        self,
        method_name: str,
        *args,
        mask: AgentMask | None = None,
        return_results: Literal[False] = False,
        inplace: bool = True,
        **kwargs,
    ) -> Self: ...

    @overload
    def do(
        self,
        method_name: str,
        *args,
        mask: AgentMask | None = None,
        return_results: Literal[True],
        inplace: bool = True,
        **kwargs,
    ) -> Any: ...

    def do(
        self,
        method_name: str,
        *args,
        mask: AgentMask | None = None,
        return_results: bool = False,
        inplace: bool = True,
        **kwargs,
    ) -> Self | Any:
        masked_df = self._get_masked_df(mask)
        # If the mask is empty, we can use the object as is
        if len(masked_df) == len(self._df):
            obj = self._get_obj(inplace)
            method = getattr(obj, method_name)
            result = method(*args, **kwargs)
        else:  # If the mask is not empty, we need to create a new masked AbstractAgentSet and concatenate the AbstractAgentSets at the end
            obj = self._get_obj(inplace=False)
            obj._df = masked_df
            original_masked_index = obj._get_obj_copy(obj.index)
            method = getattr(obj, method_name)
            result = method(*args, **kwargs)
            obj._concatenate_agentsets(
                [self],
                duplicates_allowed=True,
                keep_first_only=True,
                original_masked_index=original_masked_index,
            )
            if inplace:
                for key, value in obj.__dict__.items():
                    setattr(self, key, value)
                obj = self
        if return_results:
            return result
        else:
            return obj

    def get(
        self,
        attr_names: IntoExpr | Iterable[IntoExpr] | None,
        mask: AgentPolarsMask = None,
    ) -> pl.Series | pl.DataFrame:
        masked_df = self._get_masked_df(mask)
        if attr_names is None:
            # Return all columns except unique_id
            return masked_df.select(pl.exclude("unique_id"))
        attr_names = self.df.select(attr_names).columns.copy()
        if not attr_names:
            return masked_df
        masked_df = masked_df.select(attr_names)
        if masked_df.shape[1] == 1:
            return masked_df[masked_df.columns[0]]
        return masked_df

    def remove(self, agents: PolarsIdsLike | AgentMask, inplace: bool = True) -> Self:
        if isinstance(agents, str) and agents == "active":
            agents = self.active_agents
        if agents is None or (isinstance(agents, Iterable) and len(agents) == 0):
            return self._get_obj(inplace)
        agents = self._df_index(self._get_masked_df(agents), "unique_id")
        sets = self.model.sets.remove(agents, inplace=inplace)
        for agentset in sets.df.keys():
            if isinstance(agentset, self.__class__):
                return agentset
        return self

    def set(
        self,
        attr_names: str | Collection[str] | dict[str, Any] | None = None,
        values: Any | None = None,
        mask: AgentPolarsMask = None,
        inplace: bool = True,
    ) -> Self:
        obj = self._get_obj(inplace)
        masked_df = obj._get_masked_df(mask)

        if not attr_names:
            attr_names = masked_df.columns
            attr_names.remove("unique_id")

        def process_single_attr(
            masked_df: pl.DataFrame, attr_name: str, values: Any
        ) -> pl.DataFrame:
            if isinstance(values, pl.DataFrame):
                values_series = values.to_series()
            elif isinstance(values, (pl.Expr, pl.Series, Collection)):
                values_series = pl.Series(values)
            else:
                values_series = pl.repeat(values, len(masked_df))
            return masked_df.with_columns(values_series.alias(attr_name))

        if isinstance(attr_names, str) and values is not None:
            masked_df = process_single_attr(masked_df, attr_names, values)
        elif isinstance(attr_names, Collection) and values is not None:
            if isinstance(values, Collection) and len(attr_names) == len(values):
                for attribute, val in zip(attr_names, values):
                    masked_df = process_single_attr(masked_df, attribute, val)
            else:
                for attribute in attr_names:
                    masked_df = process_single_attr(masked_df, attribute, values)
        elif isinstance(attr_names, dict):
            for key, val in attr_names.items():
                masked_df = process_single_attr(masked_df, key, val)
        else:
            raise ValueError(
                "attr_names must be a string, a collection of string or a dictionary with columns as keys and values."
            )
        unique_id_column = None
        if "unique_id" not in obj._df:
            unique_id_column = self._generate_unique_ids(len(masked_df)).alias(
                "unique_id"
            )
            obj._df = obj._df.with_columns(unique_id_column)
            masked_df = masked_df.with_columns(unique_id_column)
        b_mask = obj._get_bool_mask(mask)
        non_masked_df = obj._df.filter(b_mask.not_())
        original_index = obj._df.select("unique_id")
        obj._df = pl.concat([non_masked_df, masked_df], how="diagonal_relaxed")
        obj._df = original_index.join(obj._df, on="unique_id", how="left")
        obj._update_mask(original_index["unique_id"], unique_id_column)
        return obj

    def select(
        self,
        mask: AgentPolarsMask = None,
        filter_func: Callable[[Self], pl.Series] | None = None,
        n: int | None = None,
        negate: bool = False,
        inplace: bool = True,
    ) -> Self:
        obj = self._get_obj(inplace)
        mask = obj._get_bool_mask(mask)
        if filter_func:
            mask = mask & filter_func(obj)
        if n is not None:
            mask = (obj._df["unique_id"]).is_in(
                obj._df.filter(mask).sample(n)["unique_id"]
            )
        if negate:
            mask = mask.not_()
        obj._mask = mask
        return obj

    def shuffle(self, inplace: bool = True) -> Self:
        obj = self._get_obj(inplace)
        obj._df = obj._df.sample(
            fraction=1,
            shuffle=True,
            seed=obj.random.integers(np.iinfo(np.int32).max),
        )
        return obj

    def sort(
        self,
        by: str | Sequence[str],
        ascending: bool | Sequence[bool] = True,
        inplace: bool = True,
        **kwargs,
    ) -> Self:
        obj = self._get_obj(inplace)
        if isinstance(ascending, bool):
            descending = not ascending
        else:
            descending = [not a for a in ascending]
        obj._df = obj._df.sort(by=by, descending=descending, **kwargs)
        return obj

    def _concatenate_agentsets(
        self,
        agentsets: Iterable[Self],
        duplicates_allowed: bool = True,
        keep_first_only: bool = True,
        original_masked_index: pl.Series | None = None,
    ) -> Self:
        if not duplicates_allowed:
            indices_list = [self._df["unique_id"]] + [
                agentset._df["unique_id"] for agentset in agentsets
            ]
            all_indices = pl.concat(indices_list)
            if all_indices.is_duplicated().any():
                raise ValueError(
                    "Some ids are duplicated in the AgentSets that are trying to be concatenated"
                )
        if duplicates_allowed & keep_first_only:
            # Find the original_index list (ie longest index list), to sort correctly the rows after concatenation
            max_length = max(len(agentset) for agentset in agentsets)
            for agentset in agentsets:
                if len(agentset) == max_length:
                    original_index = agentset._df["unique_id"]
            final_dfs = [self._df]
            final_active_indices = [self._df["unique_id"]]
            final_indices = self._df["unique_id"].clone()
            for obj in iter(agentsets):
                # Remove agents that are already in the final DataFrame
                final_dfs.append(
                    obj._df.filter(pl.col("unique_id").is_in(final_indices).not_())
                )
                # Add the indices of the active agents of current AgentSet
                final_active_indices.append(obj._df.filter(obj._mask)["unique_id"])
                # Update the indices of the agents in the final DataFrame
                final_indices = pl.concat(
                    [final_indices, final_dfs[-1]["unique_id"]], how="vertical"
                )
            # Left-join original index with concatenated dfs to keep original ids order
            final_df = original_index.to_frame().join(
                pl.concat(final_dfs, how="diagonal_relaxed"), on="unique_id", how="left"
            )
            #
            final_active_index = pl.concat(final_active_indices, how="vertical")

        else:
            final_df = pl.concat([obj._df for obj in agentsets], how="diagonal_relaxed")
            final_active_index = pl.concat(
                [obj._df.filter(obj._mask)["unique_id"] for obj in agentsets]
            )
        final_mask = final_df["unique_id"].is_in(final_active_index)
        self._df = final_df
        self._mask = final_mask
        # If some ids were removed in the do-method, we need to remove them also from final_df
        if not isinstance(original_masked_index, type(None)):
            ids_to_remove = original_masked_index.filter(
                original_masked_index.is_in(self._df["unique_id"]).not_()
            )
            if not ids_to_remove.is_empty():
                self.remove(ids_to_remove, inplace=True)
        return self

    def _get_bool_mask(
        self,
        mask: AgentPolarsMask = None,
    ) -> pl.Series | pl.Expr:
        def bool_mask_from_series(mask: pl.Series) -> pl.Series:
            if (
                isinstance(mask, pl.Series)
                and mask.dtype == pl.Boolean
                and len(mask) == len(self._df)
            ):
                return mask
            return self._df["unique_id"].is_in(mask)

        if isinstance(mask, pl.Expr):
            return mask
        elif isinstance(mask, pl.Series):
            return bool_mask_from_series(mask)
        elif isinstance(mask, pl.DataFrame):
            if "unique_id" in mask.columns:
                return bool_mask_from_series(mask["unique_id"])
            elif len(mask.columns) == 1 and mask.dtypes[0] == pl.Boolean:
                return bool_mask_from_series(mask[mask.columns[0]])
            else:
                raise KeyError(
                    "DataFrame must have a 'unique_id' column or a single boolean column."
                )
        elif mask is None or mask == "all":
            return pl.repeat(True, len(self._df))
        elif mask == "active":
            return self._mask
        elif isinstance(mask, Collection):
            return bool_mask_from_series(pl.Series(mask, dtype=pl.UInt64))
        else:
            return bool_mask_from_series(pl.Series([mask], dtype=pl.UInt64))

    def _get_masked_df(
        self,
        mask: AgentPolarsMask = None,
    ) -> pl.DataFrame:
        if (isinstance(mask, pl.Series) and mask.dtype == pl.Boolean) or isinstance(
            mask, pl.Expr
        ):
            return self._df.filter(mask)
        elif isinstance(mask, pl.DataFrame):
            if not mask["unique_id"].is_in(self._df["unique_id"]).all():
                raise KeyError(
                    "Some 'unique_id' of mask are not present in DataFrame 'unique_id'."
                )
            return mask.select("unique_id").join(self._df, on="unique_id", how="left")
        elif isinstance(mask, pl.Series):
            if not mask.is_in(self._df["unique_id"]).all():
                raise KeyError(
                    "Some 'unique_id' of mask are not present in DataFrame 'unique_id'."
                )
            mask_df = mask.to_frame("unique_id")
            return mask_df.join(self._df, on="unique_id", how="left")
        elif mask is None or mask == "all":
            return self._df
        elif mask == "active":
            return self._df.filter(self._mask)
        else:
            if isinstance(mask, Collection):
                mask_series = pl.Series(mask, dtype=pl.UInt64)
            else:
                mask_series = pl.Series([mask], dtype=pl.UInt64)
            if not mask_series.is_in(self._df["unique_id"]).all():
                raise KeyError(
                    "Some 'unique_id' of mask are not present in DataFrame 'unique_id'."
                )
            mask_df = mask_series.to_frame("unique_id")
            return mask_df.join(self._df, on="unique_id", how="left")

    @overload
    def _get_obj_copy(self, obj: pl.Series) -> pl.Series: ...

    @overload
    def _get_obj_copy(self, obj: pl.DataFrame) -> pl.DataFrame: ...

    def _get_obj_copy(self, obj: pl.Series | pl.DataFrame) -> pl.Series | pl.DataFrame:
        return obj.clone()

    def _discard(self, ids: PolarsIdsLike) -> Self:
        mask = self._get_bool_mask(ids)

        if isinstance(self._mask, pl.Series):
            original_active_indices = self._df.filter(self._mask)["unique_id"]

        self._df = self._df.filter(mask.not_())

        if isinstance(self._mask, pl.Series):
            self._update_mask(original_active_indices)

        return self

    def _update_mask(
        self, original_active_indices: pl.Series, new_indices: pl.Series | None = None
    ) -> None:
        if new_indices is not None:
            self._mask = self._df["unique_id"].is_in(
                original_active_indices
            ) | self._df["unique_id"].is_in(new_indices)
        else:
            self._mask = self._df["unique_id"].is_in(original_active_indices)

    def __getattr__(self, key: str) -> Any:
        if key == "name":
            return self.name
        super().__getattr__(key)
        return self._df[key]

    def _generate_unique_ids(self, n: int) -> pl.Series:
        return pl.Series(
            self.random.integers(1, np.iinfo(np.uint64).max, size=n, dtype=np.uint64)
        )

    @overload
    def __getitem__(
        self,
        key: str | tuple[AgentPolarsMask, str],
    ) -> pl.Series: ...

    @overload
    def __getitem__(
        self,
        key: (
            AgentPolarsMask
            | Collection[str]
            | tuple[
                AgentPolarsMask,
                Collection[str],
            ]
        ),
    ) -> pl.DataFrame: ...

    def __getitem__(
        self,
        key: (
            str
            | Collection[str]
            | AgentPolarsMask
            | tuple[AgentPolarsMask, str]
            | tuple[
                AgentPolarsMask,
                Collection[str],
            ]
        ),
    ) -> pl.Series | pl.DataFrame:
        attr = super().__getitem__(key)
        assert isinstance(attr, (pl.Series, pl.DataFrame))
        return attr

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self._df.iter_rows(named=True))

    def __len__(self) -> int:
        return len(self._df)

    def __reversed__(self) -> Iterator:
        return reversed(iter(self._df.iter_rows(named=True)))

    @property
    def df(self) -> pl.DataFrame:
        return self._df

    @df.setter
    def df(self, agents: pl.DataFrame) -> None:
        if "unique_id" not in agents.columns:
            raise KeyError("DataFrame must have a unique_id column.")
        self._df = agents

    @property
    def active_agents(self) -> pl.DataFrame:
        return self.df.filter(self._mask)

    @active_agents.setter
    def active_agents(self, mask: AgentPolarsMask) -> None:
        self.select(mask=mask, inplace=True)

    @property
    def inactive_agents(self) -> pl.DataFrame:
        return self.df.filter(~self._mask)

    @property
    def index(self) -> pl.Series:
        return self._df["unique_id"]

    @property
    def pos(self) -> pl.DataFrame:
        return super().pos

    @property
    def name(self) -> str:
        """Return the name of the AgentSet."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Set the name of the AgentSet."""
        self._name = value
