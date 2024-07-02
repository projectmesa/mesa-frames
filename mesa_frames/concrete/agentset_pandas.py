from collections.abc import Callable, Collection, Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, Self, overload

import pandas as pd
import polars as pl

from mesa_frames.abstract.agents import AgentSetDF
from mesa_frames.concrete.agentset_polars import AgentSetPolars
from mesa_frames.types import PandasIdsLike, PandasMaskLike

if TYPE_CHECKING:
    from mesa_frames.concrete.model import ModelDF


class AgentSetPandas(AgentSetDF):
    _agents: pd.DataFrame
    _mask: pd.Series
    _copy_with_method: dict[str, tuple[str, list[str]]] = {
        "_agents": ("copy", ["deep"]),
        "_mask": ("copy", ["deep"]),
    }
    """A pandas-based implementation of the AgentSet.

    Attributes
    ----------
    _agents : pd.DataFrame
        The agents in the AgentSet.
    _copy_only_reference : list[str] = ['_model']
        A list of attributes to copy with a reference only.
    _copy_with_method: dict[str, tuple[str, list[str]]] = {
        "_agents": ("copy", ["deep"]),
        "_mask": ("copy", ["deep"]),
    }
        A dictionary of attributes to copy with a specified method and arguments.
    _mask : pd.Series
        A boolean mask indicating which agents are active.
    _model : ModelDF
        The model that the AgentSetDF belongs to.

    Properties
    ----------
    active_agents(self) -> pd.DataFrame
        Get the active agents in the AgentSetPandas.
    agents(self) -> pd.DataFrame
        Get or set the agents in the AgentSetPandas.
    inactive_agents(self) -> pd.DataFrame
        Get the inactive agents in the AgentSetPandas.
    model(self) -> ModelDF
        Get the model associated with the AgentSetPandas.
    random(self) -> Generator
        Get the random number generator associated with the model.

    Methods
    -------
    __init__(self, model: ModelDF) -> None
        Initialize a new AgentSetPandas.
    add(self, other: pd.DataFrame | Sequence[Any] | dict[str, Any], inplace: bool = True) -> Self
        Add agents to the AgentSetPandas.
    contains(self, ids: PandasIdsLike) -> bool | pd.Series
        Check if agents with the specified IDs are in the AgentSetPandas.
    copy(self, deep: bool = False, memo: dict | None = None) -> Self
        Create a copy of the AgentSetPandas.
    discard(self, ids: PandasIdsLike, inplace: bool = True) -> Self
        Remove an agent from the AgentSetPandas. Does not raise an error if the agent is not found.
    do(self, method_name: str, *args, return_results: bool = False, inplace: bool = True, **kwargs) -> Self | Any
        Invoke a method on the AgentSetPandas.
    get(self, attr_names: str | Collection[str] | None, mask: PandasMaskLike = None) -> pd.Series | pd.DataFrame
        Retrieve the value of a specified attribute for each agent in the AgentSetPandas.
    remove(self, ids: PandasIdsLike, inplace: bool = True) -> Self
        Remove agents from the AgentSetPandas.
    select(self, mask: PandasMaskLike = None, filter_func: Callable[[Self], PandasMaskLike] | None = None, n: int | None = None, negate: bool = False, inplace: bool = True) -> Self
        Select agents in the AgentSetPandas based on the given criteria.
    set(self, attr_names: str | Collection[str] | dict[str, Any] | None = None, values: Any | None = None, mask: PandasMaskLike | None = None, inplace: bool = True) -> Self
        Set the value of a specified attribute or attributes for each agent in the mask in the AgentSetPandas.
    shuffle(self, inplace: bool = True) -> Self
        Shuffle the order of agents in the AgentSetPandas.
    sort(self, by: str | Sequence[str], ascending: bool | Sequence[bool] = True, inplace: bool = True, **kwargs) -> Self
        Sort the agents in the AgentSetPandas based on the given criteria.
    to_polars(self) -> "AgentSetPolars"
        Convert the AgentSetPandas to an AgentSetPolars.
    _get_bool_mask(self, mask: PandasMaskLike = None) -> pd.Series
        Get a boolean mask for selecting agents.
    _get_masked_df(self, mask: PandasMaskLike = None) -> pd.DataFrame
        Get a DataFrame of agents that match the mask.
    __getattr__(self, key: str) -> pd.Series
        Retrieve an attribute of the underlying DataFrame.
    __iter__(self) -> Iterator
        Get an iterator for the agents in the AgentSetPandas.
    __len__(self) -> int
        Get the number of agents in the AgentSetPandas.
    __repr__(self) -> str
        Get the string representation of the AgentSetPandas.
    __reversed__(self) -> Iterator
        Get a reversed iterator for the agents in the AgentSetPandas.
    __str__(self) -> str
        Get the string representation of the AgentSetPandas.
    """

    def __init__(self, model: "ModelDF") -> None:
        self._model = model
        self._agents = (
            pd.DataFrame(columns=["unique_id"])
            .astype({"unique_id": "int64"})
            .set_index("unique_id")
        )
        self._mask = pd.Series(True, index=self._agents.index, dtype=pd.BooleanDtype())

    def add(
        self,
        agents: pd.DataFrame | Sequence[Any] | dict[str, Any],
        inplace: bool = True,
    ) -> Self:
        obj = self._get_obj(inplace)
        if isinstance(agents, pd.DataFrame):
            new_agents = agents
            if "unique_id" != agents.index.name:
                try:
                    new_agents.set_index("unique_id", inplace=True, drop=True)
                except KeyError:
                    raise KeyError("DataFrame must have a unique_id column/index.")
        elif isinstance(agents, dict):
            if "unique_id" not in agents:
                raise KeyError("Dictionary must have a unique_id key.")
            index = agents.pop("unique_id")
            if not isinstance(index, list):
                index = [index]
            new_agents = pd.DataFrame(agents, index=pd.Index(index, name="unique_id"))
        else:
            if len(agents) != len(obj._agents.columns) + 1:
                raise ValueError(
                    "Length of data must match the number of columns in the AgentSet if being added as a Collection."
                )
            columns = pd.Index(["unique_id"]).append(obj._agents.columns.copy())
            new_agents = pd.DataFrame([agents], columns=columns).set_index(
                "unique_id", drop=True
            )

        if new_agents.index.dtype != "int64":
            raise TypeError("unique_id must be of type int64.")

        if not obj._agents.index.intersection(new_agents.index).empty:
            raise KeyError("Some IDs already exist in the agent set.")

        original_active_indices = obj._mask.index[obj._mask].copy()

        obj._agents = pd.concat([obj._agents, new_agents])

        obj._update_mask(original_active_indices, new_agents.index)

        return obj

    @overload
    def contains(self, ids: int) -> bool: ...

    @overload
    def contains(self, ids: PandasIdsLike) -> pd.Series: ...

    def contains(self, ids: PandasIdsLike) -> bool | pd.Series:
        if isinstance(ids, pd.Series):
            return ids.isin(self._agents.index)
        elif isinstance(ids, pd.Index):
            return pd.Series(
                ids.isin(self._agents.index), index=ids, dtype=pd.BooleanDtype()
            )
        elif isinstance(ids, Collection):
            return pd.Series(list(ids), index=list(ids)).isin(self._agents.index)
        else:
            return ids in self._agents.index

    def get(
        self,
        attr_names: str | Collection[str] | None = None,
        mask: PandasMaskLike = None,
    ) -> pd.Index | pd.Series | pd.DataFrame:
        mask = self._get_bool_mask(mask)
        if attr_names is None:
            return self._agents.loc[mask]
        else:
            if attr_names == "unique_id":
                return self._agents.loc[mask].index
            if isinstance(attr_names, str):
                return self._agents.loc[mask, attr_names]
            if isinstance(attr_names, Collection):
                return self._agents.loc[mask, list(attr_names)]

    def remove(
        self,
        ids: PandasIdsLike,
        inplace: bool = True,
    ) -> Self:
        obj = self._get_obj(inplace)
        initial_len = len(obj._agents)
        mask = obj._get_bool_mask(ids)
        remove_ids = obj._agents[mask].index
        original_active_indices = obj._mask.index[obj._mask].copy()
        obj._agents.drop(remove_ids, inplace=True)
        if len(obj._agents) == initial_len:
            raise KeyError("Some IDs were not found in agent set.")

        self._update_mask(original_active_indices)
        return obj

    def set(
        self,
        attr_names: str | dict[str, Any] | Collection[str] | None = None,
        values: Any | None = None,
        mask: PandasMaskLike = None,
        inplace: bool = True,
    ) -> Self:
        obj = self._get_obj(inplace)
        b_mask = obj._get_bool_mask(mask)
        masked_df = obj._get_masked_df(mask)

        if not attr_names:
            attr_names = masked_df.columns

        if isinstance(attr_names, dict):
            for key, val in attr_names.items():
                masked_df.loc[:, key] = val
        elif (
            isinstance(attr_names, str)
            or (
                isinstance(attr_names, Collection)
                and all(isinstance(n, str) for n in attr_names)
            )
        ) and values is not None:
            if not isinstance(attr_names, str):  # isinstance(attr_names, Collection)
                attr_names = list(attr_names)
            masked_df.loc[:, attr_names] = values
        else:
            raise ValueError(
                "Either attr_names must be a dictionary with columns as keys and values or values must be provided."
            )

        non_masked_df = obj._agents[~b_mask]
        original_index = obj._agents.index
        obj._agents = pd.concat([non_masked_df, masked_df])
        obj._agents = obj._agents.reindex(original_index)
        return obj

    def select(
        self,
        mask: PandasMaskLike = None,
        filter_func: Callable[[Self], PandasMaskLike] | None = None,
        n: int | None = None,
        negate: bool = False,
        inplace: bool = True,
    ) -> Self:
        obj = self._get_obj(inplace)
        bool_mask = obj._get_bool_mask(mask)
        if filter_func:
            bool_mask = bool_mask & obj._get_bool_mask(filter_func(obj))
        if negate:
            bool_mask = ~bool_mask
        if n is not None:
            bool_mask = pd.Series(
                obj._agents.index.isin(obj._agents[bool_mask].sample(n).index),
                index=obj._agents.index,
            )
        obj._mask = bool_mask
        return obj

    def shuffle(self, inplace: bool = True) -> Self:
        obj = self._get_obj(inplace)
        obj._agents = obj._agents.sample(frac=1)
        return obj

    def sort(
        self,
        by: str | Sequence[str],
        ascending: bool | Sequence[bool] = True,
        inplace: bool = True,
        **kwargs,
    ) -> Self:
        obj = self._get_obj(inplace)
        obj._agents.sort_values(by=by, ascending=ascending, **kwargs, inplace=True)
        return obj

    def to_polars(self) -> AgentSetPolars:
        new_obj = AgentSetPolars(self._model)
        new_obj._agents = pl.DataFrame(self._agents)
        new_obj._mask = pl.Series(self._mask)
        return new_obj

    def _concatenate_agentsets(
        self,
        agentsets: Iterable[Self],
        duplicates_allowed: bool = True,
        keep_first_only: bool = True,
        original_masked_index: pd.Index | None = None,
    ) -> Self:
        if not duplicates_allowed:
            indices = [self._agents.index.to_series()] + [
                agentset._agents.index.to_series() for agentset in agentsets
            ]
            pd.concat(indices, verify_integrity=True)
        if duplicates_allowed & keep_first_only:
            final_df = self._agents.copy()
            final_mask = self._mask.copy()
            for obj in iter(agentsets):
                final_df = final_df.combine_first(obj._agents)
                final_mask = final_mask.combine_first(obj._mask)
        else:
            final_df = pd.concat([obj._agents for obj in agentsets])
            final_mask = pd.concat([obj._mask for obj in agentsets])
        self._agents = final_df
        self._mask = final_mask
        if not isinstance(original_masked_index, type(None)):
            ids_to_remove = original_masked_index.difference(self._agents.index)
            if not ids_to_remove.empty:
                self.remove(ids_to_remove, inplace=True)
        return self

    def _get_bool_mask(
        self,
        mask: PandasMaskLike = None,
    ) -> pd.Series:
        if isinstance(mask, pd.Series) and mask.dtype == bool:
            return mask
        elif isinstance(mask, pd.DataFrame):
            return pd.Series(
                self._agents.index.isin(mask.index), index=self._agents.index
            )
        elif isinstance(mask, list):
            return pd.Series(self._agents.index.isin(mask), index=self._agents.index)
        elif mask is None or mask == "all":
            return pd.Series(True, index=self._agents.index)
        elif mask == "active":
            return self._mask
        else:
            return pd.Series(self._agents.index.isin([mask]), index=self._agents.index)

    def _get_masked_df(
        self,
        mask: PandasMaskLike = None,
    ) -> pd.DataFrame:
        if isinstance(mask, pd.Series) and mask.dtype == bool:
            return self._agents.loc[mask]
        elif isinstance(mask, pd.DataFrame):
            if not mask.index.isin(self._agents.index).all():
                raise KeyError(
                    "Some 'unique_id' of mask are not present in DataFrame 'unique_id'."
                )
            if mask.index.name != "unique_id":
                if "unique_id" in mask.columns:
                    mask.set_index("unique_id", inplace=True, drop=True)
                else:
                    raise KeyError("DataFrame must have a unique_id column/index.")
            return pd.DataFrame(index=mask.index).join(
                self._agents, on="unique_id", how="left"
            )
        elif isinstance(mask, pd.Series):
            if not mask.isin(self._agents.index).all():
                raise KeyError(
                    "Some 'unique_id' of mask are not present in DataFrame 'unique_id'."
                )
            mask_df = mask.to_frame("unique_id").set_index("unique_id")
            return mask_df.join(self._agents, on="unique_id", how="left")
        elif mask is None or mask == "all":
            return self._agents
        elif mask == "active":
            return self._agents.loc[self._mask]
        else:
            mask_series = pd.Series(mask)
            if not mask_series.isin(self._agents.index).all():
                raise KeyError(
                    "Some 'unique_id' of mask are not present in DataFrame 'unique_id'."
                )
            mask_df = mask_series.to_frame("unique_id").set_index("unique_id")
            return mask_df.join(self._agents, on="unique_id", how="left")

    @overload
    def _get_obj_copy(self, obj: pd.Series) -> pd.Series: ...

    @overload
    def _get_obj_copy(self, obj: pd.DataFrame) -> pd.DataFrame: ...

    @overload
    def _get_obj_copy(self, obj: pd.Index) -> pd.Index: ...

    def _get_obj_copy(
        self, obj: pd.Series | pd.DataFrame | pd.Index
    ) -> pd.Series | pd.DataFrame | pd.Index:
        return obj.copy()

    def _update_mask(
        self,
        original_active_indices: pd.Index,
        new_active_indices: pd.Index | None = None,
    ) -> None:
        # Update the mask with the old active agents and the new agents
        if new_active_indices is None:
            self._mask = pd.Series(
                self._agents.index.isin(original_active_indices),
                index=self._agents.index,
                dtype=pd.BooleanDtype(),
            )
        else:
            self._mask = pd.Series(
                self._agents.index.isin(original_active_indices)
                | self._agents.index.isin(new_active_indices),
                index=self._agents.index,
                dtype=pd.BooleanDtype(),
            )

    def __getattr__(self, name: str) -> Any:
        super().__getattr__(name)
        return getattr(self._agents, name)

    def __iter__(self) -> Iterator:
        return iter(self._agents.iterrows())

    def __len__(self) -> int:
        return len(self._agents)

    def __reversed__(self) -> Iterator:
        return iter(self._agents[::-1].iterrows())

    @property
    def agents(self) -> pd.DataFrame:
        return self._agents

    @agents.setter
    def agents(self, new_agents: pd.DataFrame) -> None:
        if new_agents.index.name == "unique_id":
            pass
        elif "unique_id" in new_agents.columns:
            new_agents.set_index("unique_id", inplace=True, drop=True)
        else:
            raise KeyError("The DataFrame should have a 'unique_id' index/column")
        self._agents = new_agents

    @property
    def active_agents(self) -> pd.DataFrame:
        return self._agents.loc[self._mask]

    @active_agents.setter
    def active_agents(self, mask: PandasMaskLike) -> None:
        self.select(mask=mask, inplace=True)

    @property
    def inactive_agents(self) -> pd.DataFrame:
        return self._agents.loc[~self._mask]

    @property
    def index(self) -> pd.Index:
        return self._agents.index
