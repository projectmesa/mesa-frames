import polars as pl

from .base.agent import AgentSetDF
from .base.model import ModelDF


class AgentSetPolars(AgentSetDF):
    _agents: pl.DataFrame
    _mask: pl.Expr | pl.Series
    """A polars-based implementation of the AgentSet."""

    def __init__(self, model: ModelDF):
        """Create a new AgentSetDF.

        Parameters
        ----------
        model : ModelDF
            The model to which the AgentSetDF belongs.

        Attributes
        ----------
        agents : pl.DataFrame
            The agents in the AgentSetDF.
        model : ModelDF
            The model to which the AgentSetDF belongs.
        """
        super().__init__(model)
        self._agents = pl.DataFrame(schema={"unique_id": pl.String})
        self._mask = pl.repeat(True, len(self.agents))

    @property
    def agents(self) -> pl.DataFrame:
        if self._agents is None:
            self._agents = pl.DataFrame(schema={"unique_id": pl.String})
        return self._agents

    @property
    def active_agents(self) -> pl.DataFrame:
        return self.agents.filter(self._mask)

    @property
    def inactive_agents(self) -> pl.DataFrame:
        return self.agents.filter(~self._mask)

    def select(
        self,
        mask: pl.Expr | pl.Series | pl.DataFrame | None = None,
        filter_func: Callable[[Self], pl.Series] | None = None,
        n: int = 0,
    ) -> Self:
        if mask is None:  # if not mask doesn't work
            mask = pl.repeat(True, len(self.agents))
        elif isinstance(mask, pl.DataFrame):
            mask = self.agents["unique_id"].is_in(mask["unique_id"])
        if filter_func:
            mask = mask & filter_func(self)
        if n != 0:
            mask = (
                self.agents.filter(mask)
                .sample(n)["unique_id"]
                .is_in(self.agents["unique_id"])
            )
        self._mask = mask
        return self

    def shuffle(self) -> Self:
        self.agents = self.agents.sample(fraction=1)
        return self

    def sort(
        self,
        by: IntoExpr | Iterable[IntoExpr],
        *more_by: IntoExpr,
        descending: bool | Sequence[bool] = False,
        nulls_last: bool = False,
    ) -> Self:
        """Sort the agents in the agent set based on the given criteria.

        Parameters
        ----------
        by (IntoExpr | Iterable[IntoExpr]): The attribute(s) to sort by.
        more_by (IntoExpr): Additional attributes to sort by.
        descending (bool | Sequence[bool]): Whether to sort in descending order.
        nulls_last (bool): Whether to place null values last.

        Returns
        ----------
        AgentSetDF: The sorted agent set.
        """
        self.agents = self.agents.sort(
            by=by, *more_by, descending=descending, nulls_last=nulls_last
        )
        return self

    def get_attribute(self, attr_names: str) -> pl.Series:
        return self.agents.filter(self._mask)[attr_names]

    def set_attribute(self, attr_names: str, value: Any) -> Self:
        if type(value) == pl.Series:
            self.agents.filter(self._mask).with_columns(**{attr_names: value})
        else:
            self.agents.filter(self._mask).with_columns(**{attr_names: pl.lit(value)})
        return self

    def add(
        self,
        n: int,
        data: FrameInitTypes | None = None,
        schema: SchemaDefinition | None = None,
        schema_overrides: SchemaDict | None = None,
        orient: Orientation | None = None,
        infer_schema_length: int | None = N_INFER_DEFAULT,
        nan_to_null: bool = False,
    ) -> Self:
        """Adds new agents to the agent set.

        Parameters
        ----------
        n : int
            The number of agents to add.
        data : dict, Sequence, ndarray, Series, or pandas.DataFrame
            Two-dimensional data in various forms; dict input must contain Sequences, Generators, or a range. Sequence may contain Series or other Sequences.
        schema : Sequence of str, (str,DataType) pairs, or a {str:DataType,} dict
            The DataFrame schema may be declared in several ways:
            - As a dict of {name:type} pairs; if type is None, it will be auto-inferred.
            - As a list of column names; in this case types are automatically inferred.
            - As a list of (name,type) pairs; this is equivalent to the dictionary form.
            If you supply a list of column names that does not match the names in the underlying data, the names given here will overwrite them. The number of names given in the schema should match the underlying data dimensions.
        schema_overrides : dict, default None
            Support type specification or override of one or more columns; note that any dtypes inferred from the schema param will be overridden.
            The number of entries in the schema should match the underlying data dimensions, unless a sequence of dictionaries is being passed, in which case a *partial* schema can be declared to prevent specific fields from being loaded.
        orient : {'col', 'row'}, default None
            Whether to interpret two-dimensional data as columns or as rows.
            If None, the orientation is inferred by matching the columns and data dimensions.
            If this does not yield conclusive results, column orientation is used.
        infer_schema_length : int or None
            The maximum number of rows to scan for schema inference. If set to None, the full data may be scanned *(this is slow)*.
            This parameter only applies if the input data is a sequence or generator of rows; other input is read as-is.
        nan_to_null :  bool, default False
            If the data comes from one or more numpy arrays, can optionally convert input data np.nan values to null instead. This is a no-op for all other input data.

        Returns
        ----------
        AgentSetPolars: The updated agent set.
        """
        new_df = pl.DataFrame(
            data=data,
            schema=schema,
            schema_overrides=schema_overrides,
            orient=orient,
            infer_schema_length=infer_schema_length,
            nan_to_null=nan_to_null,
        )

        if "unique_id" not in new_df.columns:
            new_df = new_df.with_columns(
                unique_id=pl.Series(
                    values=self.random.random(n) * 10**8, dtype=pl.Int64
                )
            )

        old_active_agents = self.agents.filter(self._mask)["unique_id"]
        self.agents = pl.concat([self.agents, new_df])
        self._mask = self.agents["unique_id"].is_in(old_active_agents) | self.agents[
            "unique_id"
        ].is_in(new_df["unique_id"])
        return self

    def discard(self, id: int) -> Self:
        with suppress(KeyError):
            self.agents = self.agents.filter(self.agents["unique_id"] != id)
        return self

    def remove(self, id: int) -> Self:
        self.agents = self.agents.filter(self.agents["unique_id"] != id)
        return self


class AgentsPolars(AgentsDF):
    agentsets: list[AgentSetPolars]
    """A polars implementation of a collection of AgentSetDF. All agents of the model are stored here."""

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
    def active_agents(self) -> pl.DataFrame:
        return pl.concat([agentset.active_agents for agentset in self.agentsets])

    @property
    def inactive_agents(self) -> pl.DataFrame:
        return pl.concat([agentset.inactive_agents for agentset in self.agentsets])

    def select(
        self,
        mask: pl.Expr | pl.Series | pl.DataFrame | None = None,
        filter_func: Callable[[AgentSetDF], pl.Series] | None = None,
        n: int = 0,
    ) -> Self:
        n, r = int(n / len(self.agentsets)), n % len(self.agentsets)
        new_agentsets: list[AgentSetPolars] = []
        for agentset in self.agentsets:
            if mask is None:
                agentset_mask = mask
            elif isinstance(mask, pl.DataFrame):
                agentset_mask = agentset.agents["unique_id"].is_in(mask["unique_id"])
            elif isinstance(mask, pl.Series):
                agentset_mask = agentset.agents["unique_id"].is_in(mask)
            agentset.select(mask=agentset_mask, filter_func=filter_func, n=n + r)
            if len(agentset.active_agents) > n:
                r = len(agentset.active_agents) - n
            new_agentsets.append(agentset)
        self.agentsets = new_agentsets
        return self

    def get_attribute(self, attr_names: str) -> pl.Series:
        return pl.concat(
            [agentset.get_attribute(attr_names) for agentset in self.agentsets]
        )

    def add(self, agentsets: AgentSetPolars | list[AgentSetPolars]) -> Self:
        return super().add(agentsets)  # type: ignore #child classes are not checked?
