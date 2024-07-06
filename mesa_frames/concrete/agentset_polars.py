from collections.abc import Callable, Collection, Iterable, Iterator, Sequence
from typing import TYPE_CHECKING

import polars as pl
from polars._typing import IntoExpr
from typing_extensions import Any, Self, overload

from mesa_frames.concrete.agents import AgentSetDF
from mesa_frames.types import PolarsIdsLike, PolarsMaskLike

if TYPE_CHECKING:
    from mesa_frames.concrete.agentset_pandas import AgentSetPandas
    from mesa_frames.concrete.model import ModelDF


class AgentSetPolars(AgentSetDF):
    _agents: pl.DataFrame
    _copy_with_method: dict[str, tuple[str, list[str]]] = {
        "_agents": ("clone", []),
    }
    _copy_only_reference: list[str] = ["_model", "_mask"]
    _mask: pl.Expr | pl.Series

    """A polars-based implementation of the AgentSet.

    Attributes
    ----------
    _agents : pl.DataFrame
        The agents in the AgentSet.
    _copy_only_reference : list[str] = ["_model", "_mask"]
        A list of attributes to copy with a reference only.
    _copy_with_method: dict[str, tuple[str, list[str]]] = {
        "_agents": ("copy", ["deep"]),
        "_mask": ("copy", ["deep"]),
    }
        A dictionary of attributes to copy with a specified method and arguments.
    model : ModelDF
        The model to which the AgentSet belongs.
    _mask : pl.Series
        A boolean mask indicating which agents are active.

    Properties
    ----------
    active_agents(self) -> pl.DataFrame
        Get the active agents in the AgentSetPolars.
    agents(self) -> pl.DataFrame
        Get or set the agents in the AgentSetPolars.
    inactive_agents(self) -> pl.DataFrame
        Get the inactive agents in the AgentSetPolars.
    model(self) -> ModelDF
        Get the model associated with the AgentSetPolars.
    random(self) -> Generator
        Get the random number generator associated with the model.


    Methods
    -------
    __init__(self, model: ModelDF) -> None
        Initialize a new AgentSetPolars.
    add(self, other: pl.DataFrame | Sequence[Any] | dict[str, Any], inplace: bool = True) -> Self
        Add agents to the AgentSetPolars.
    contains(self, ids: PolarsIdsLike) -> bool | pl.Series
        Check if agents with the specified IDs are in the AgentSetPolars.
    copy(self, deep: bool = False, memo: dict | None = None) -> Self
        Create a copy of the AgentSetPolars.
    discard(self, ids: PolarsIdsLike, inplace: bool = True) -> Self
        Remove an agent from the AgentSetPolars. Does not raise an error if the agent is not found.
    do(self, method_name: str, *args, return_results: bool = False, inplace: bool = True, **kwargs) -> Self | Any
        Invoke a method on the AgentSetPolars.
    get(self, attr_names: IntoExpr | Iterable[IntoExpr] | None, mask: PolarsMaskLike = None) -> pl.Series | pl.DataFrame
        Retrieve the value of a specified attribute for each agent in the AgentSetPolars.
    remove(self, ids: PolarsIdsLike, inplace: bool = True) -> Self
        Remove agents from the AgentSetPolars.
    select(self, mask: PolarsMaskLike = None, filter_func: Callable[[Self], PolarsMaskLike] | None = None, n: int | None = None, negate: bool = False, inplace: bool = True) -> Self
        Select agents in the AgentSetPolars based on the given criteria.
    set(self, attr_names: str | Collection[str] | dict[str, Any] | None = None, values: Any | None = None, mask: PolarsMaskLike | None = None, inplace: bool = True) -> Self
        Set the value of a specified attribute or attributes for each agent in the mask in the AgentSetPolars.
    shuffle(self, inplace: bool = True) -> Self
        Shuffle the order of agents in the AgentSetPolars.
    sort(self, by: str | Sequence[str], ascending: bool | Sequence[bool] = True, inplace: bool = True, **kwargs) -> Self
        Sort the agents in the AgentSetPolars based on the given criteria.
    to_pandas(self) -> "AgentSetPandas"
        Convert the AgentSetPolars to an AgentSetPandas.
    _get_bool_mask(self, mask: PolarsMaskLike = None) -> pl.Series | pl.Expr
        Get a boolean mask for selecting agents.
    _get_masked_df(self, mask: PolarsMaskLike = None) -> pl.DataFrame
        Get a DataFrame of agents that match the mask.
    __getattr__(self, key: str) -> pl.Series
        Retrieve an attribute of the underlying DataFrame.
    __iter__(self) -> Iterator
        Get an iterator for the agents in the AgentSetPolars.
    __len__(self) -> int
        Get the number of agents in the AgentSetPolars.
    __repr__(self) -> str
        Get the string representation of the AgentSetPolars.
    __reversed__(self) -> Iterator
        Get a reversed iterator for the agents in the AgentSetPolars.
    __str__(self) -> str
        Get the string representation of the AgentSetPolars.

    """

    def __init__(self, model: "ModelDF") -> None:
        """Initialize a new AgentSetPolars.

        Parameters
        ----------
        model : ModelDF
            The model that the agent set belongs to.

        Returns
        -------
        None
        """
        self._model = model
        self._agents = pl.DataFrame(schema={"unique_id": pl.Int64})
        self._mask = pl.repeat(True, len(self._agents), dtype=pl.Boolean, eager=True)

    def add(
        self,
        agents: pl.DataFrame | Sequence[Any] | dict[str, Any],
        inplace: bool = True,
    ) -> Self:
        """Add agents to the AgentSetPolars.

        Parameters
        ----------
        other : pl.DataFrame | Sequence[Any] | dict[str, Any]
            The agents to add.
        inplace : bool, optional
            Whether to add the agents in place, by default True.

        Returns
        -------
        Self
            The updated AgentSetPolars.
        """
        obj = self._get_obj(inplace)
        if isinstance(agents, pl.DataFrame):
            if "unique_id" not in agents.columns:
                raise KeyError("DataFrame must have a unique_id column.")
            new_agents = agents
        elif isinstance(agents, dict):
            if "unique_id" not in agents:
                raise KeyError("Dictionary must have a unique_id key.")
            new_agents = pl.DataFrame(agents)
        else:
            if len(agents) != len(obj._agents.columns):
                raise ValueError(
                    "Length of data must match the number of columns in the AgentSet if being added as a Collection."
                )
            new_agents = pl.DataFrame([agents], schema=obj._agents.schema)

        if new_agents["unique_id"].dtype != pl.Int64:
            raise TypeError("unique_id column must be of type int64.")

        # If self._mask is pl.Expr, then new mask is the same.
        # If self._mask is pl.Series[bool], then new mask has to be updated.

        if isinstance(obj._mask, pl.Series):
            original_active_indices = obj._agents.filter(obj._mask)["unique_id"]

        obj._agents = pl.concat([obj._agents, new_agents], how="diagonal_relaxed")

        if isinstance(obj._mask, pl.Series):
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
            return agents.is_in(self._agents["unique_id"])
        elif isinstance(agents, Collection):
            return pl.Series(agents).is_in(self._agents["unique_id"])
        else:
            return agents in self._agents["unique_id"]

    def get(
        self,
        attr_names: IntoExpr | Iterable[IntoExpr] | None,
        mask: PolarsMaskLike = None,
    ) -> pl.Series | pl.DataFrame:
        masked_df = self._get_masked_df(mask)
        attr_names = self.agents.select(attr_names).columns.copy()
        if not attr_names:
            return masked_df
        masked_df = masked_df.select(attr_names)
        if masked_df.shape[1] == 1:
            return masked_df[masked_df.columns[0]]
        return masked_df

    def remove(self, ids: PolarsIdsLike, inplace: bool = True) -> Self:
        obj = self._get_obj(inplace=inplace)
        initial_len = len(obj._agents)
        mask = obj._get_bool_mask(ids)

        if isinstance(obj._mask, pl.Series):
            original_active_indices = obj._agents.filter(obj._mask)["unique_id"]

        obj._agents = obj._agents.filter(mask.not_())
        if len(obj._agents) == initial_len:
            raise KeyError(f"IDs {ids} not found in agent set.")

        if isinstance(obj._mask, pl.Series):
            obj._update_mask(original_active_indices)
        return obj

    def set(
        self,
        attr_names: str | Collection[str] | dict[str, Any] | None = None,
        values: Any | None = None,
        mask: PolarsMaskLike = None,
        inplace: bool = True,
    ) -> Self:
        obj = self._get_obj(inplace)
        b_mask = obj._get_bool_mask(mask)
        masked_df = obj._get_masked_df(mask)

        if not attr_names:
            attr_names = masked_df.columns
            attr_names.remove("unique_id")

        def process_single_attr(
            masked_df: pl.DataFrame, attr_name: str, values: Any
        ) -> pl.DataFrame:
            if isinstance(values, pl.DataFrame):
                return masked_df.with_columns(values.to_series().alias(attr_name))
            elif isinstance(values, pl.Expr):
                return masked_df.with_columns(values.alias(attr_name))
            if isinstance(values, pl.Series):
                return masked_df.with_columns(values.alias(attr_name))
            else:
                if isinstance(values, Collection):
                    values = pl.Series(values)
                else:
                    values = pl.repeat(values, len(masked_df))
                return masked_df.with_columns(values.alias(attr_name))

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
        non_masked_df = obj._agents.filter(b_mask.not_())
        original_index = obj._agents.select("unique_id")
        obj._agents = pl.concat([non_masked_df, masked_df], how="diagonal_relaxed")
        obj._agents = original_index.join(obj._agents, on="unique_id", how="left")
        return obj

    def select(
        self,
        mask: PolarsMaskLike = None,
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
            mask = (obj._agents["unique_id"]).is_in(
                obj._agents.filter(mask).sample(n)["unique_id"]
            )
        if negate:
            mask = mask.not_()
        obj._mask = mask
        return obj

    def shuffle(self, inplace: bool = True) -> Self:
        obj = self._get_obj(inplace)
        obj._agents = obj._agents.sample(fraction=1, shuffle=True)
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
        obj._agents = obj._agents.sort(by=by, descending=descending, **kwargs)
        return obj

    def to_pandas(self) -> "AgentSetPandas":
        from mesa_frames.concrete.agentset_pandas import AgentSetPandas

        new_obj = AgentSetPandas(self._model)
        new_obj._agents = self._agents.to_pandas()
        if isinstance(self._mask, pl.Series):
            new_obj._mask = self._mask.to_pandas()
        else:  # self._mask is Expr
            new_obj._mask = (
                self._agents["unique_id"]
                .is_in(self._agents.filter(self._mask)["unique_id"])
                .to_pandas()
            )
        return new_obj

    def _concatenate_agentsets(
        self,
        agentsets: Iterable[Self],
        duplicates_allowed: bool = True,
        keep_first_only: bool = True,
        original_masked_index: pl.Series | None = None,
    ) -> Self:
        if not duplicates_allowed:
            indices_list = [self._agents["unique_id"]] + [
                agentset._agents["unique_id"] for agentset in agentsets
            ]
            all_indices = pl.concat(indices_list)
            if all_indices.is_duplicated().any():
                raise ValueError(
                    "Some ids are duplicated in the AgentSetDFs that are trying to be concatenated"
                )
        if duplicates_allowed & keep_first_only:
            # Find the original_index list (ie longest index list), to sort correctly the rows after concatenation
            max_length = max(len(agentset) for agentset in agentsets)
            for agentset in agentsets:
                if len(agentset) == max_length:
                    original_index = agentset._agents["unique_id"]
            final_dfs = [self._agents]
            final_active_indices = [self._agents["unique_id"]]
            final_indices = self._agents["unique_id"].clone()
            for obj in iter(agentsets):
                # Remove agents that are already in the final DataFrame
                final_dfs.append(
                    obj._agents.filter(pl.col("unique_id").is_in(final_indices).not_())
                )
                # Add the indices of the active agents of current AgentSet
                final_active_indices.append(obj._agents.filter(obj._mask)["unique_id"])
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
            final_df = pl.concat(
                [obj._agents for obj in agentsets], how="diagonal_relaxed"
            )
            final_active_index = pl.concat(
                [obj._agents.filter(obj._mask)["unique_id"] for obj in agentsets]
            )
        final_mask = final_df["unique_id"].is_in(final_active_index)
        self._agents = final_df
        self._mask = final_mask
        # If some ids were removed in the do-method, we need to remove them also from final_df
        if not isinstance(original_masked_index, type(None)):
            ids_to_remove = original_masked_index.filter(
                original_masked_index.is_in(self._agents["unique_id"]).not_()
            )
            if not ids_to_remove.is_empty():
                self.remove(ids_to_remove, inplace=True)
        return self

    def _get_bool_mask(
        self,
        mask: PolarsMaskLike = None,
    ) -> pl.Series | pl.Expr:
        def bool_mask_from_series(mask: pl.Series) -> pl.Series:
            if (
                isinstance(mask, pl.Series)
                and mask.dtype == pl.Boolean
                and len(mask) == len(self._agents)
            ):
                return mask
            return self._agents["unique_id"].is_in(mask)

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
            return pl.repeat(True, len(self._agents))
        elif mask == "active":
            return self._mask
        elif isinstance(mask, Collection):
            return bool_mask_from_series(pl.Series(mask))
        else:
            return bool_mask_from_series(pl.Series([mask]))

    def _get_masked_df(
        self,
        mask: PolarsMaskLike = None,
    ) -> pl.DataFrame:
        if (isinstance(mask, pl.Series) and mask.dtype == pl.Boolean) or isinstance(
            mask, pl.Expr
        ):
            return self._agents.filter(mask)
        elif isinstance(mask, pl.DataFrame):
            if not mask["unique_id"].is_in(self._agents["unique_id"]).all():
                raise KeyError(
                    "Some 'unique_id' of mask are not present in DataFrame 'unique_id'."
                )
            return mask.select("unique_id").join(
                self._agents, on="unique_id", how="left"
            )
        elif isinstance(mask, pl.Series):
            if not mask.is_in(self._agents["unique_id"]).all():
                raise KeyError(
                    "Some 'unique_id' of mask are not present in DataFrame 'unique_id'."
                )
            mask_df = mask.to_frame("unique_id")
            return mask_df.join(self._agents, on="unique_id", how="left")
        elif mask is None or mask == "all":
            return self._agents
        elif mask == "active":
            return self._agents.filter(self._mask)
        else:
            if isinstance(mask, Collection):
                mask_series = pl.Series(mask)
            else:
                mask_series = pl.Series([mask])
            if not mask_series.is_in(self._agents["unique_id"]).all():
                raise KeyError(
                    "Some 'unique_id' of mask are not present in DataFrame 'unique_id'."
                )
            mask_df = mask_series.to_frame("unique_id")
            return mask_df.join(self._agents, on="unique_id", how="left")

    @overload
    def _get_obj_copy(self, obj: pl.Series) -> pl.Series: ...

    @overload
    def _get_obj_copy(self, obj: pl.DataFrame) -> pl.DataFrame: ...

    def _get_obj_copy(self, obj: pl.Series | pl.DataFrame) -> pl.Series | pl.DataFrame:
        return obj.clone()

    def _update_mask(
        self, original_active_indices: pl.Series, new_indices: pl.Series | None = None
    ) -> None:
        if new_indices is not None:
            self._mask = self._agents["unique_id"].is_in(
                original_active_indices
            ) | self._agents["unique_id"].is_in(new_indices)
        else:
            self._mask = self._agents["unique_id"].is_in(original_active_indices)

    def __getattr__(self, key: str) -> pl.Series:
        super().__getattr__(key)
        return self._agents[key]

    @overload
    def __getitem__(
        self,
        key: str | tuple[PolarsMaskLike, str],
    ) -> pl.Series: ...

    @overload
    def __getitem__(
        self,
        key: (
            PolarsMaskLike
            | Collection[str]
            | tuple[
                PolarsMaskLike,
                Collection[str],
            ]
        ),
    ) -> pl.DataFrame: ...

    def __getitem__(
        self,
        key: (
            str
            | Collection[str]
            | PolarsMaskLike
            | tuple[PolarsMaskLike, str]
            | tuple[
                PolarsMaskLike,
                Collection[str],
            ]
        ),
    ) -> pl.Series | pl.DataFrame:
        attr = super().__getitem__(key)
        assert isinstance(attr, (pl.Series, pl.DataFrame))
        return attr

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self._agents.iter_rows(named=True))

    def __len__(self) -> int:
        return len(self._agents)

    def __reversed__(self) -> Iterator:
        return reversed(iter(self._agents.iter_rows(named=True)))

    @property
    def agents(self) -> pl.DataFrame:
        return self._agents

    @agents.setter
    def agents(self, agents: pl.DataFrame) -> None:
        if "unique_id" not in agents.columns:
            raise KeyError("DataFrame must have a unique_id column.")
        self._agents = agents

    @property
    def active_agents(self) -> pl.DataFrame:
        return self.agents.filter(self._mask)

    @active_agents.setter
    def active_agents(self, mask: PolarsMaskLike) -> None:
        self.select(mask=mask, inplace=True)

    @property
    def inactive_agents(self) -> pl.DataFrame:
        return self.agents.filter(~self._mask)

    @property
    def index(self) -> pl.Series:
        return self._agents["unique_id"]
