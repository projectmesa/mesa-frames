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
from mesa_frames.concrete._update_masked import _MaskedUpdateMixin
from mesa_frames.concrete.mixin import PolarsMixin
from mesa_frames.types_ import AgentMask, AgentPolarsMask, IntoExpr, PolarsIdsLike
from mesa_frames.utils import copydoc
import mesa_frames


@copydoc(AbstractAgentSet)
class AgentSet(_MaskedUpdateMixin, AbstractAgentSet, PolarsMixin):
    """Polars-based implementation of AgentSet."""

    _df: pl.DataFrame
    _copy_with_method: dict[str, tuple[str, list[str]]] = {
        "_df": ("clone", []),
    }
    _copy_only_reference: list[str] = ["_model", "_mask"]
    _mask: pl.Expr | pl.Series

    def __init__(
        self, model: mesa_frames.concrete.model.Model, name: str | None = None
    ) -> None:
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
        self._uid_cache_df_id: int | None = None
        self._uid_sorted_cache: np.ndarray | None = None
        self._uid_sort_idx_cache: np.ndarray | None = None
        self._mask = pl.repeat(True, len(self._df), dtype=pl.Boolean, eager=True)

    def _invalidate_uid_cache(self) -> None:
        self._uid_cache_df_id = None
        self._uid_sorted_cache = None
        self._uid_sort_idx_cache = None

    def _get_sorted_uids(self) -> tuple[np.ndarray, np.ndarray]:
        df_id = id(self._df)
        if (
            self._uid_cache_df_id == df_id
            and self._uid_sorted_cache is not None
            and self._uid_sort_idx_cache is not None
        ):
            return self._uid_sorted_cache, self._uid_sort_idx_cache

        uids = self._df["unique_id"].to_numpy()
        sort_idx = np.argsort(uids)
        sorted_uids = uids[sort_idx]
        self._uid_cache_df_id = df_id
        self._uid_sorted_cache = sorted_uids
        self._uid_sort_idx_cache = sort_idx
        return sorted_uids, sort_idx

    def rename(self, new_name: str, inplace: bool = True) -> Self:
        """Rename this agent set. If attached to AgentSetRegistry, delegate for uniqueness enforcement.

        Parameters
        ----------
        new_name : str
            Desired new name.

        inplace : bool, optional
            Whether to perform the rename in place. If False, a renamed copy is
            returned, by default True.

        Returns
        -------
        Self
            The updated AgentSet (or a renamed copy when ``inplace=False``).

        Raises
        ------
        ValueError
            If name conflicts occur and delegate encounters errors.
        """
        # Respect inplace semantics consistently with other mutators
        obj = self._get_obj(inplace)

        # Always delegate to the container's accessor if available through the model's sets
        # Check if we have a model and can find the AgentSetRegistry that contains this set
        try:
            if self in self.model.sets:
                # Track the index of this set so we can retrieve the renamed copy even
                # when the registry canonicalizes the requested name.
                target_idx = next(
                    i for i, aset in enumerate(self.model.sets) if aset is self
                )
                reg = self.model.sets.rename(self, new_name, inplace=inplace)
                if inplace:
                    return self
                return reg[target_idx]
        except KeyError:
            pass

        # Fall back to local rename if isn't found in a an AgentSetRegistry
        obj._name = new_name
        return obj

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

        if isinstance(obj._mask, pl.Series):
            if originally_empty:
                # When starting from an empty AgentSet, the initial mask is a
                # zero-length boolean Series; expand it to match the new rows.
                obj._mask = pl.repeat(True, len(obj._df), dtype=pl.Boolean, eager=True)
            else:
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
        obj = self._get_obj(inplace)
        # Normalize to Series of unique_ids
        ids = obj._df_index(obj._get_masked_df(agents), "unique_id")
        # Validate presence
        if not ids.is_in(obj._df["unique_id"]).all():
            raise KeyError("Some 'unique_id' of mask are not present in this AgentSet.")
        # Remove by ids
        return obj._discard(ids)

    def update(
        self,
        target: PolarsIdsLike | pl.DataFrame | dict[str, object] | None = None,
        updates: dict[str, object] | None = None,
        *,
        mask: AgentMask | np.ndarray | None = None,
        backend: Literal["auto", "polars"] = "auto",
        mask_col: str | None = None,
    ) -> None:
        """Update agent attributes.

        Parameters
        ----------
        target : PolarsIdsLike | pl.DataFrame | dict[str, object] | None, optional
            Optional id selector (scalar/sequence/Series) or a DataFrame with a
            ``unique_id`` column.
        updates : dict[str, object] | None, optional
            Mapping of column names to update values.
        mask : AgentMask | np.ndarray | None, optional
            Optional selector limiting which agents are updated.
        backend : Literal["auto", "polars"], optional
            Accepted for API compatibility; only Polars paths are used.
        mask_col : str | None, optional
            When ``mask``/``target`` is a DataFrame, optional name of a boolean
            column to interpret as per-row selector.
        """
        if updates is None and isinstance(target, dict):
            updates = target
            target = None

        if updates is None or len(updates) == 0:
            raise ValueError("update() requires a non-empty updates dict")
        self._reject_callables(updates)
        if backend not in {"auto", "polars"}:
            raise ValueError('backend must be one of: "auto", "polars"')
        if target is not None and mask is not None:
            raise ValueError("Provide either target or mask, not both")

        selector: object = target if target is not None else mask
        if selector is None:
            selector = "all"

        # Bootstrapping: allow creating rows on an empty AgentSet.
        if self._df.is_empty() and selector == "all":
            if any(isinstance(v, pl.Expr) for v in updates.values()):
                raise ValueError(
                    "Cannot initialize an empty AgentSet using pl.Expr updates"
                )
            if any(isinstance(v, str) for v in updates.values()):
                raise ValueError(
                    "Cannot initialize an empty AgentSet using copy-from-column updates"
                )

            lengths: list[int] = []
            for v in updates.values():
                if isinstance(v, pl.Series):
                    lengths.append(int(v.len()))
                elif isinstance(v, np.ndarray):
                    lengths.append(int(v.shape[0]))
                elif isinstance(v, (list, tuple)):
                    lengths.append(len(v))

            if not lengths:
                raise ValueError(
                    "Cannot initialize an empty AgentSet from scalar-only updates"
                )
            n = lengths[0]
            if any(l != n for l in lengths):
                raise ValueError("Update value lengths must match when initializing")

            init_data: dict[str, object] = {}
            for col, v in updates.items():
                if isinstance(v, pl.Series):
                    if int(v.len()) != n:
                        raise ValueError("Series length mismatch")
                    init_data[col] = v
                elif isinstance(v, np.ndarray):
                    if int(v.shape[0]) != n:
                        raise ValueError("Array length mismatch")
                    init_data[col] = v
                elif isinstance(v, (list, tuple)):
                    if len(v) != n:
                        raise ValueError("Sequence length mismatch")
                    init_data[col] = list(v)
                else:
                    init_data[col] = [v] * n

            self.add(pl.DataFrame(init_data), inplace=True)
            return

        if isinstance(selector, pl.Expr):
            raise TypeError(
                "update(mask=pl.Expr) is not supported; pass ids, a boolean mask, or a string mask"
            )

        mask_bool = self._mask_to_bool(selector, mask_col=mask_col)

        # Support passing values aligned to the *selection* (n_selected) instead
        # of the full AgentSet height (n_total), without requiring joins.
        #
        # Important: when the selector is an id-like collection, we align values
        # by unique_id order, not by boolean-mask df-order.
        n_total = int(self._df.height)
        selected_idx_df_order = np.flatnonzero(mask_bool)
        n_selected = int(selected_idx_df_order.shape[0])

        selector_ids: object | None = None
        if isinstance(selector, np.ndarray):
            if selector.dtype != bool:
                selector_ids = selector
        elif isinstance(selector, pl.Series):
            if selector.dtype != pl.Boolean:
                selector_ids = selector
        elif isinstance(selector, pl.DataFrame):
            if "unique_id" in selector.columns:
                ids = selector["unique_id"]
                if mask_col is not None:
                    if mask_col not in selector.columns:
                        raise KeyError(f"mask_col not found in DataFrame: {mask_col}")
                    ids = selector.filter(pl.col(mask_col))["unique_id"]
                selector_ids = ids
        elif selector is None or (
            isinstance(selector, str) and selector in {"all", "active"}
        ):
            selector_ids = None
        elif isinstance(selector, Collection) and not isinstance(
            selector, (str, bytes)
        ):
            selector_ids = selector
        else:
            selector_ids = [selector]

        row_idx_from_ids: np.ndarray | None = None
        if selector_ids is not None:
            ids_arr = np.asarray(selector_ids)
            if ids_arr.ndim == 0:
                ids_arr = ids_arr.reshape(1)
            if int(ids_arr.shape[0]) == n_selected:
                sorted_uids, sort_idx = self._get_sorted_uids()
                pos = np.searchsorted(sorted_uids, ids_arr)
                if (pos >= sorted_uids.shape[0]).any():
                    raise KeyError("One or more unique_id values not present")
                found = sorted_uids[pos] == ids_arr
                if not bool(np.all(found)):
                    raise KeyError("One or more unique_id values not present")
                row_idx_from_ids = sort_idx[pos]

        def _vector_len(value: object) -> int | None:
            if isinstance(value, pl.Series):
                return int(value.len())
            if isinstance(value, np.ndarray):
                if value.ndim != 1:
                    raise ValueError("ndarray update values must be 1-D")
                return int(value.shape[0])
            if isinstance(value, (list, tuple)):
                return len(value)
            return None

        def _expand_to_full(value: object, *, fill_idx: np.ndarray) -> np.ndarray:
            if isinstance(value, pl.Series):
                arr = value.to_numpy()
            elif isinstance(value, np.ndarray):
                arr = value
            elif isinstance(value, (list, tuple)):
                arr = np.asarray(value)
            else:  # pragma: no cover
                raise TypeError("unsupported vector update value")

            if arr.dtype.kind in {"b", "i", "u", "f"}:
                full = np.zeros(n_total, dtype=arr.dtype)
            else:
                full = np.empty(n_total, dtype=arr.dtype)
                if n_total:
                    full[:] = "" if arr.dtype.kind in {"U", "S"} else None

            if n_selected:
                full[fill_idx] = arr
            return full

        if n_selected and updates:
            expanded: dict[str, object] = {}
            for col, value in updates.items():
                v_len = _vector_len(value)
                if v_len is not None and v_len == n_selected and v_len != n_total:
                    fill_idx = (
                        row_idx_from_ids
                        if row_idx_from_ids is not None
                        else selected_idx_df_order
                    )
                    expanded[col] = _expand_to_full(value, fill_idx=fill_idx)
                else:
                    expanded[col] = value
            updates = expanded

        self._df = self._apply_masked_updates(self._df, mask_bool, updates)
        self._invalidate_uid_cache()

    def _mask_to_bool(self, mask: object, *, mask_col: str | None = None) -> np.ndarray:
        n_total = int(len(self._df))

        if mask is None or (isinstance(mask, str) and mask == "all"):
            return np.ones(n_total, dtype=bool)

        if isinstance(mask, str) and mask == "active":
            if isinstance(self._mask, pl.Series):
                if int(self._mask.len()) != n_total:
                    raise ValueError("Active mask length mismatch")
                return self._mask.to_numpy().astype(bool, copy=False)
            raise TypeError("active mask is not available as a boolean Series")

        if isinstance(mask, np.ndarray):
            if mask.dtype == bool:
                if mask.ndim != 1 or int(mask.shape[0]) != n_total:
                    raise ValueError("Boolean mask ndarray length mismatch")
                return mask.astype(bool, copy=False)
            return (
                self._df["unique_id"]
                .is_in(pl.Series(mask))
                .to_numpy()
                .astype(bool, copy=False)
            )

        if isinstance(mask, pl.Series):
            if mask.dtype == pl.Boolean:
                if int(mask.len()) != n_total:
                    raise ValueError("Boolean mask Series length mismatch")
                return mask.to_numpy().astype(bool, copy=False)
            return self._df["unique_id"].is_in(mask).to_numpy().astype(bool, copy=False)

        if isinstance(mask, pl.DataFrame):
            if "unique_id" in mask.columns:
                ids = mask["unique_id"]
                if mask_col is not None:
                    if mask_col not in mask.columns:
                        raise KeyError(f"mask_col not found in DataFrame: {mask_col}")
                    ids = mask.filter(pl.col(mask_col))["unique_id"]
                return (
                    self._df["unique_id"].is_in(ids).to_numpy().astype(bool, copy=False)
                )
            if len(mask.columns) == 1 and mask.dtypes[0] == pl.Boolean:
                s = mask[mask.columns[0]]
                if int(s.len()) != n_total:
                    raise ValueError("Boolean mask DataFrame length mismatch")
                return s.to_numpy().astype(bool, copy=False)
            raise KeyError(
                "DataFrame mask must have a 'unique_id' column or a single boolean column"
            )

        if isinstance(mask, Collection) and not isinstance(mask, (str, bytes)):
            return (
                self._df["unique_id"]
                .is_in(pl.Series(mask))
                .to_numpy()
                .astype(bool, copy=False)
            )

        return (
            self._df["unique_id"]
            .is_in(pl.Series([mask]))
            .to_numpy()
            .astype(bool, copy=False)
        )

    def lookup(
        self,
        target: PolarsIdsLike | pl.DataFrame,
        columns: list[str] | None = None,
        *,
        as_df: bool = True,
    ) -> pl.DataFrame | dict[str, np.ndarray] | np.ndarray:
        """Fetch rows by unique_id without joins."""
        if isinstance(target, pl.DataFrame):
            if "unique_id" not in target.columns:
                raise KeyError("AgentSet.lookup target DataFrame must have 'unique_id'")
            ids = target["unique_id"]
        else:
            ids = target

        ids_arr = np.asarray(ids)
        if ids_arr.ndim == 0:
            ids_arr = ids_arr.reshape(1)

        sorted_uids, sort_idx = self._get_sorted_uids()
        pos = np.searchsorted(sorted_uids, ids_arr)
        if (pos >= sorted_uids.shape[0]).any():
            raise KeyError("One or more unique_id values not present")
        found = sorted_uids[pos] == ids_arr
        if not bool(np.all(found)):
            raise KeyError("One or more unique_id values not present")
        row_idx = sort_idx[pos]

        out = self._df[row_idx]
        if columns is not None:
            out = out.select(columns)

        if as_df:
            return out
        if columns is not None and len(columns) == 1:
            return out[columns[0]].to_numpy()
        return {col: out[col].to_numpy() for col in out.columns}

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
        obj._invalidate_uid_cache()
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
        obj._invalidate_uid_cache()
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
        self._invalidate_uid_cache()
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
        elif mask is None or (isinstance(mask, str) and mask == "all"):
            return pl.repeat(True, len(self._df))
        elif isinstance(mask, str) and mask == "active":
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
        elif mask is None or (isinstance(mask, str) and mask == "all"):
            return self._df
        elif isinstance(mask, str) and mask == "active":
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
        self._invalidate_uid_cache()

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
        # Avoid interpreting special/protocol attributes as dataframe columns.
        if key.startswith("__") and key.endswith("__"):
            raise AttributeError(key)
        try:
            return self._df[key]
        except (pl.exceptions.ColumnNotFoundError, KeyError) as exc:
            raise AttributeError(key) from exc

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
        self._invalidate_uid_cache()

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
        self.rename(value)
