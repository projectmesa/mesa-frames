"""
Concrete implementations of mesa-frames components.

This package contains the concrete implementations of the abstract base classes
defined in mesa_frames.abstract. It provides ready-to-use classes for building
agent-based models using DataFrame-based storage, with support for both pandas
and Polars backends.

Subpackages:
    pandas: Contains pandas-based implementations of agent sets, mixins, and spatial structures.
    polars: Contains Polars-based implementations of agent sets, mixins, and spatial structures.

Modules:
    agents: Defines the AgentsDF class, a collection of AgentSetDFs.
    model: Provides the ModelDF class, the base class for models in mesa-frames.

Classes:
    From pandas.agentset:
        AgentSetPandas(AgentSetDF, PandasMixin): A pandas-based implementation of the AgentSet.

    From pandas.mixin:
        PandasMixin(DataFrameMixin): A pandas-based implementation of DataFrame operations.

    From pandas.space:
        GridPandas(GridDF, PandasMixin): A pandas-based implementation of Grid.

    From polars subpackage:
        Similar classes as in the pandas subpackage, but using Polars as the backend.

    From agents:
        AgentsDF(AgentContainer): A collection of AgentSetDFs. All agents of the model are stored here.

    From model:
        ModelDF: Base class for models in the mesa-frames library.

Usage:
    Users can import the concrete implementations directly from this package:

    from mesa_frames.concrete import ModelDF, AgentsDF
    from mesa_frames.concrete.pandas import AgentSetPandas, GridPandas

    # For Polars-based implementations
    from mesa_frames.concrete.polars import AgentSetPolars, GridPolars

    class MyModel(ModelDF):
        def __init__(self):
            super().__init__()
            self.agents.add(AgentSetPandas(self))
            self.space = GridPandas(self, dimensions=[10, 10])
            # ... other initialization code

Note:
    The choice between pandas and Polars implementations depends on the user's
    preference and performance requirements. Both provide similar functionality
    but may have different performance characteristics depending on the specific
    use case.

For more detailed information on each class, refer to their respective module
and class docstrings.
"""
