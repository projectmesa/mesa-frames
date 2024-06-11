import pandas as pd

from .base import AgentSetDF


class AgentSetPandas(AgentSetDF):
    _agents: pd.DataFrame
    _mask: pd.Series[bool]
    """A pandas-based implementation of the AgentSet.

    Attributes
    ----------
    model : ModelDF
        The model to which the AgentSet belongs.
    _mask : pd.Series[bool]
        A boolean mask indicating which agents are active.
    _agents : pd.DataFrame
        The agents in the AgentSet.
    _skip_copy : list[str]
        A list of attributes to skip during the copy process.
    """

    def __new__(cls, model: ModelDF) -> Self:
        obj = super().__new__(cls, model)
        obj._agents = pd.DataFrame(columns=["unique_id"]).set_index("unique_id")
        obj._mask = pd.Series(True, index=obj._agents.index)
        return obj

    def __contains__(self, id: Hashable) -> bool:
        return id in self._agents.index

    def __deepcopy__(self, memo: dict) -> Self:
        obj = super().__deepcopy__(memo)
        obj._agents = self._agents.copy(deep=True)
        return obj

    def __iter__(self):
        return self._agents.iterrows()

    def __reversed__(self) -> Iterable:
        return self._agents[::-1].iterrows()

    @property
    def agents(self) -> pd.DataFrame:
        return self._agents

    @property
    def active_agents(self) -> pd.DataFrame:
        return self._agents.loc[self._mask]

    @active_agents.setter  # When a property is overriden, so it is the getter
    def active_agents(self, mask: PandasMaskLike) -> None:
        return AgentContainer.active_agents.fset(self, mask)  # type: ignore

    @property
    def inactive_agents(self) -> pd.DataFrame:
        return self._agents.loc[~self._mask]

    def _get_bool_mask(
        self,
        mask: PandasMaskLike | None = None,
    ) -> pd.Series:
        if isinstance(mask, pd.Series) and mask.dtype == bool:
            return mask
        elif isinstance(mask, self.__class__):
            return pd.Series(
                self._agents.index.isin(mask.agents.index), index=self._agents.index
            )
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

    def copy(
        self,
        deep: bool = False,
        skip: list[str] | str | None = None,
        memo: dict | None = None,
    ) -> Self:
        obj = super().copy(deep, skip, memo)
        obj._agents = self._agents.copy(deep=deep)
        obj._mask = self._mask.copy(deep=deep)
        return obj

    def select(
        self,
        mask: PandasMaskLike | None = None,
        filter_func: Callable[[Self], PandasMaskLike] | None = None,
        n: int | None = None,
        inplace: bool = True,
    ) -> Self:
        obj = self._get_obj(inplace)
        bool_mask = obj._get_bool_mask(mask)
        if n != None:
            bool_mask = pd.Series(
                obj._agents.index.isin(obj._agents[bool_mask].sample(n).index),
                index=obj._agents.index,
            )
        if filter_func:
            bool_mask = bool_mask & obj._get_bool_mask(filter_func(obj))
        obj._mask = bool_mask
        return obj

    def shuffle(self, inplace: bool = True) -> Self:
        obj = self._get_obj(inplace)
        obj._agents = obj._agents.sample(frac=1)
        return obj

    def sort(
        self,
        by: str | Sequence[str],
        key: ValueKeyFunc | None = None,
        ascending: bool | Sequence[bool] = True,
        inplace: bool = True,
    ) -> Self:
        """
        Sort the agents in the agent set based on the given criteria.

        Parameters
        ----------
        by : str | Sequence[str]
            The attribute(s) to sort by.
        key : ValueKeyFunc | None
            A function to use for sorting.
        ascending : bool | Sequence[bool]
            Whether to sort in ascending order.

        Returns
        ----------
        AgentSetDF: The sorted agent set.
        """
        obj = self._get_obj(inplace)
        obj._agents.sort_values(by=by, key=key, ascending=ascending, inplace=True)
        return obj

    @overload
    def set_attribute(
        self,
        attr_names: None = None,
        value: Any = Any,
        mask: PandasMaskLike = PandasMaskLike,
        inplace: bool = True,
    ) -> Self: ...

    @overload
    def set_attribute(
        self,
        attr_names: dict[str, Any],
        value: None,
        mask: PandasMaskLike | None = None,
        inplace: bool = True,
    ) -> Self: ...

    @overload
    def set_attribute(
        self,
        attr_names: str | list[str],
        value: Any,
        mask: PandasMaskLike | None = None,
        inplace: bool = True,
    ) -> Self: ...

    def set_attribute(
        self,
        attr_names: str | list[str] | dict[str, Any] | None = None,
        value: Any | None = None,
        mask: PandasMaskLike | None = None,
        inplace: bool = True,
    ) -> Self:
        obj = self._get_obj(inplace)
        mask = obj._get_bool_mask(mask)
        if attr_names is None:
            attr_names = obj._agents.columns.values.tolist()
        if isinstance(attr_names, (str, list)) and value is not None:
            obj._agents.loc[mask, attr_names] = value
        elif isinstance(attr_names, dict):
            for key, value in attr_names.items():
                obj._agents.loc[mask, key] = value
        else:
            raise ValueError(
                "attr_names must be a string or a dictionary with columns as keys and values."
            )
        return obj

    @overload
    def get_attribute(
        self,
        attr_names: list[str] | None = None,
        mask: PandasMaskLike | None = None,
    ) -> pd.DataFrame: ...

    @overload
    def get_attribute(
        self,
        attr_names: str,
        mask: PandasMaskLike | None = None,
    ) -> pd.Series: ...

    def get_attribute(
        self,
        attr_names: str | list[str] | None = None,
        mask: PandasMaskLike | None = None,
        inplace: bool = True,
    ) -> pd.Series | pd.DataFrame:
        obj = self._get_obj(inplace)
        mask = obj._get_bool_mask(mask)
        if attr_names is None:
            return obj._agents.loc[mask]
        else:
            return obj._agents.loc[mask, attr_names]

    def add(
        self,
        other: Self | pd.DataFrame | ListLike | dict[str, Any],
        inplace: bool = True,
    ) -> Self:
        obj = self._get_obj(inplace)
        if isinstance(other, obj.__class__):
            new_agents = other.agents
        elif isinstance(other, pd.DataFrame):
            new_agents = other
            if "unique_id" != other.index.name:
                try:
                    new_agents.set_index("unique_id", inplace=True, drop=True)
                except KeyError:
                    new_agents["unique_id"] = obj.random.random(len(other)) * 10**8
        elif isinstance(other, dict):
            if "unique_id" not in other:
                index = obj.random.random(len(other)) * 10**8
            if not isinstance(other["unique_id"], ListLike):
                index = [other["unique_id"]]
            else:
                index = other["unique_id"]
            new_agents = (
                pd.DataFrame(other, index=pd.Index(index))
                .reset_index(drop=True)
                .set_index("unique_id")
            )
        else:  # ListLike
            if len(other) == len(obj._agents.columns):
                # data missing unique_id
                new_agents = pd.DataFrame([other], columns=obj._agents.columns)
                new_agents["unique_id"] = obj.random.random(1) * 10**8
            elif len(other) == len(obj._agents.columns) + 1:
                new_agents = pd.DataFrame(
                    [other], columns=["unique_id"] + obj._agents.columns.values.tolist()
                )
            else:
                raise ValueError(
                    "Length of data must match the number of columns in the AgentSet if being added as a ListLike."
                )
            new_agents.set_index("unique_id", inplace=True, drop=True)
        obj._agents = pd.concat([obj._agents, new_agents])
        return obj

    def remove(self, id: PandasMaskLike, inplace: bool = True) -> Self:
        initial_len = len(self._agents)
        obj = self._get_obj(inplace)
        mask = obj._get_bool_mask(id)
        remove_ids = obj._agents[mask].index
        obj._agents.drop(remove_ids, inplace=True)
        if len(obj._agents) == initial_len:
            raise KeyError(f"IDs {id} not found in agent set.")
        return obj


class AgentsPandas(AgentsDF):
    agentsets: list[AgentSetPandas]
    """A pandas implementation of a collection of AgentSetDF. All agents of the model are stored here."""

    def __init__(self, model: ModelDF):
        """Create a new AgentsDF object.

        Parameters
        ----------
        model : ModelDF
            The model to which the AgentsDF object belongs.

        Attributes
        ----------
        agentsets : list[AgentSetDF]
            The AgentSetDFs that make up the AgentsDF object.
        model : ModelDF
            The model to which the AgentSetDF belongs.
        """
        super().__init__(model)

    @property
    def active_agents(self) -> pd.DataFrame:
        return pd.concat([agentset.active_agents for agentset in self.agentsets])

    @property
    def inactive_agents(self) -> pd.DataFrame:
        return pd.concat([agentset.inactive_agents for agentset in self.agentsets])

    def select(
        self,
        mask: pd.Series[bool] | pd.DataFrame | None = None,
        filter_func: Callable[[AgentSetDF], pd.Series[bool]] | None = None,
        n: int = 0,
    ) -> Self:
        n, r = int(n / len(self.agentsets)), n % len(self.agentsets)
        new_agentsets: list[AgentSetPandas] = []
        for agentset in self.agentsets:
            if mask is None:
                agentset_mask = mask
            elif isinstance(mask, pd.DataFrame):
                agentset_mask = pd.Series(
                    agentset.agents.index.isin(mask), index=agentset.agents.index
                )
            else:
                agentset_mask = pd.Series(
                    agentset.agents.index.isin(mask[mask].index),
                    index=agentset.agents.index,
                )
            agentset.select(mask=agentset_mask, filter_func=filter_func, n=n + r)
            if len(agentset.active_agents) > n:
                r = len(agentset.active_agents) - n
            new_agentsets.append(agentset)
        self.agentsets = new_agentsets
        return self

    def get_attribute(self, attr_names: str) -> pd.Series[Any]:
        return pd.concat(
            [agentset.get_attribute(attr_names) for agentset in self.agentsets]
        )

    def add(self, agentsets: AgentSetPandas | list[AgentSetPandas]) -> Self:
        return super().add(agentsets)  # type: ignore
