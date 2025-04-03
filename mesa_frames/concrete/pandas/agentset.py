"""
Pandas-based implementation of AgentSet for mesa-frames.

This module provides a concrete implementation of the AgentSet class using pandas
as the backend for DataFrame operations. It defines the AgentSetPandas class,
which combines the abstract AgentSetDF functionality with pandas-specific
operations for efficient agent management and manipulation.

Classes:
    AgentSetPandas(AgentSetDF, PandasMixin):
        A pandas-based implementation of the AgentSet. This class uses pandas
        DataFrames to store and manipulate agent data, providing high-performance
        operations for large numbers of agents.

The AgentSetPandas class is designed to be used within ModelDF instances or as
part of an AgentsDF collection. It leverages the power of pandas for fast and
efficient data operations on agent attributes and behaviors.

Usage:
    The AgentSetPandas class can be used directly in a model or as part of an
    AgentsDF collection:

    from mesa_frames.concrete.model import ModelDF
    from mesa_frames.concrete.pandas.agentset import AgentSetPandas
    import numpy as np

    class MyAgents(AgentSetPandas):
        def __init__(self, model):
            super().__init__(model)
            # Initialize with some agents
            self.add({'unique_id': np.arange(100), 'wealth': 10})

        def step(self):
            # Implement step behavior using pandas operations
            self.agents['wealth'] += 1

    class MyModel(ModelDF):
        def __init__(self):
            super().__init__()
            self.agents += MyAgents(self)

        def step(self):
            self.agents.step()

Note:
    This implementation relies on pandas, so users should ensure that pandas
    is installed and imported. The performance characteristics of this class
    will depend on the pandas version and the specific operations used.

For more detailed information on the AgentSetPandas class and its methods,
refer to the class docstring.
"""

import warnings
from collections.abc import Callable, Collection, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
import polars as pl
from typing_extensions import Any, Self, overload
from beartype import beartype

from mesa_frames.abstract.agents import AgentSetDF
from mesa_frames.concrete.pandas.mixin import PandasMixin
from mesa_frames.concrete.polars.agentset import AgentSetPolars
from mesa_frames.types_ import AgentPandasMask, PandasIdsLike
from mesa_frames.utils import copydoc


@beartype
@copydoc(AgentSetDF)
class AgentSetPandas(AgentSetDF, PandasMixin):
    """
    WARNING: AgentSetPandas is deprecated and will be removed in the next release of mesa-frames.

    pandas-based implementation of AgentSetDF.

    """

    _agents: pd.DataFrame
    _mask: pd.Series
    _copy_with_method: dict[str, tuple[str, list[str]]] = {
        "_agents": ("copy", ["deep"]),
        "_mask": ("copy", ["deep"]),
    }

    def __init__(self, model: "mesa_frames.concrete.model.ModelDF") -> None:
        """Initialize a new AgentSetPandas.

        Overload this method to add custom initialization logic but make sure to call super().__init__(model).

        Parameters
        ----------
        model : ModelDF
            The model associated with the AgentSetPandas.
        """
        warnings.warn(
            "AgentSetPandas is deprecated and will be removed in the next release of mesa-frames.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._model = model
        self._agents = (
            pd.DataFrame(columns=["unique_id"])
            .astype({"unique_id": "int64"})
            .set_index("unique_id")
        )
        self._mask = pd.Series(True, index=self._agents.index, dtype=pd.BooleanDtype())

    def add(  # noqa : D102
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
            new_agents.index = new_agents.index.astype("int64")

        if not obj._agents.index.intersection(new_agents.index).empty:
            raise KeyError("Some IDs already exist in the agent set.")

        original_active_indices = obj._mask.index[obj._mask].copy()

        obj._agents = pd.concat([obj._agents, new_agents])

        obj._update_mask(original_active_indices, new_agents.index)

        return obj

    @overload
    def contains(self, agents: int) -> bool: ...

    @overload
    def contains(self, agents: PandasIdsLike) -> pd.Series: ...

    def contains(self, agents: PandasIdsLike) -> bool | pd.Series:  # noqa : D102
        if isinstance(agents, pd.Series):
            return agents.isin(self._agents.index)
        elif isinstance(agents, pd.Index):
            return pd.Series(
                agents.isin(self._agents.index), index=agents, dtype=pd.BooleanDtype()
            )
        elif isinstance(agents, Collection):
            return pd.Series(list(agents), index=list(agents)).isin(self._agents.index)
        else:
            return agents in self._agents.index

    def get(  # noqa : D102
        self,
        attr_names: str | Collection[str] | None = None,
        mask: AgentPandasMask = None,
    ) -> pd.Index | pd.Series | pd.DataFrame:
        mask = self._get_bool_mask(mask)
        if attr_names is None:
            return self._agents.loc[mask]
        else:
            if isinstance(attr_names, str) and attr_names == "unique_id":
                return self._agents.loc[mask].index
            if isinstance(attr_names, str):
                return self._agents.loc[mask, attr_names]
            if isinstance(attr_names, Collection):
                return self._agents.loc[mask, list(attr_names)]

    def set(  # noqa : D102
        self,
        attr_names: str | dict[str, Any] | Collection[str] | None = None,
        values: Any | None = None,
        mask: AgentPandasMask = None,
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

    def select(  # noqa : D102
        self,
        mask: AgentPandasMask = None,
        filter_func: Callable[[Self], AgentPandasMask] | None = None,
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

    def shuffle(self, inplace: bool = True) -> Self:  # noqa : D102
        obj = self._get_obj(inplace)
        obj._agents = obj._agents.sample(
            frac=1, random_state=obj.random.integers(np.iinfo(np.int32).max)
        )
        return obj

    def sort(  # noqa : D102
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
        """Convert the AgentSetPandas to an AgentSetPolars.

        NOTE: If a methods is not backend-agnostic (i.e., it uses pandas-specific functionality), when the method is called on the Polars version of the object, it will raise an error.

        Returns
        -------
        AgentSetPolars
            An AgentSetPolars object with the same agents and active agents as the AgentSetPandas.
        """
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
        mask: AgentPandasMask = None,
    ) -> pd.Series:
        if isinstance(mask, pd.Series) and mask.dtype == bool:
            return mask
        elif isinstance(mask, pd.DataFrame):
            return pd.Series(
                self._agents.index.isin(mask.index), index=self._agents.index
            )
        elif isinstance(mask, list):
            return pd.Series(self._agents.index.isin(mask), index=self._agents.index)
        elif mask is None or isinstance(mask, str) and mask == "all":
            return pd.Series(True, index=self._agents.index)
        elif isinstance(mask, str) and mask == "active":
            return self._mask
        elif isinstance(mask, Collection):
            return pd.Series(self._agents.index.isin(mask), index=self._agents.index)
        else:
            return pd.Series(self._agents.index.isin([mask]), index=self._agents.index)

    def _get_masked_df(
        self,
        mask: AgentPandasMask = None,
    ) -> pd.DataFrame:
        if isinstance(mask, pd.Series) and mask.dtype == bool:
            return self._agents.loc[mask]
        elif isinstance(mask, pd.DataFrame):
            if mask.index.name != "unique_id":
                if "unique_id" in mask.columns:
                    mask.set_index("unique_id", inplace=True, drop=True)
                else:
                    raise KeyError("DataFrame must have a unique_id column/index.")
            return pd.DataFrame(index=mask.index).join(
                self._agents, on="unique_id", how="left"
            )
        elif isinstance(mask, pd.Series):
            mask_df = mask.to_frame("unique_id").set_index("unique_id")
            return mask_df.join(self._agents, on="unique_id", how="left")
        elif mask is None or mask == "all":
            return self._agents
        elif mask == "active":
            return self._agents.loc[self._mask]
        else:
            mask_series = pd.Series(mask)
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

    def _discard(
        self,
        ids: PandasIdsLike,
    ) -> Self:
        mask = self._get_bool_mask(ids)
        remove_ids = self._agents[mask].index
        original_active_indices = self._mask.index[self._mask].copy()
        self._agents.drop(remove_ids, inplace=True)
        self._update_mask(original_active_indices)
        return self

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

    def __getattr__(self, name: str) -> Any:  # noqa : D105
        super().__getattr__(name)
        return getattr(self._agents, name)

    def __iter__(self) -> Iterator[dict[str, Any]]:  # noqa : D105
        for index, row in self._agents.iterrows():
            row_dict = row.to_dict()
            row_dict["unique_id"] = index
            yield row_dict

    def __len__(self) -> int:  # noqa : D105
        return len(self._agents)

    def __reversed__(self) -> Iterator:  # noqa : D105
        return iter(self._agents[::-1].iterrows())

    @property
    def agents(self) -> pd.DataFrame:  # noqa : D105
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
    def active_agents(self) -> pd.DataFrame:  # noqa : D102
        return self._agents.loc[self._mask]

    @active_agents.setter
    def active_agents(self, mask: AgentPandasMask) -> None:
        self.select(mask=mask, inplace=True)

    @property
    def inactive_agents(self) -> pd.DataFrame:  # noqa : D102
        return self._agents.loc[~self._mask]

    @property
    def index(self) -> pd.Index:  # noqa : D102
        return self._agents.index

    @property
    def pos(self) -> pd.DataFrame:  # noqa : D102
        return super().pos
