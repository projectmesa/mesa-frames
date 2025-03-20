"""
Concrete implementations of mesa-frames components.

This package contains the concrete implementations of the abstract base classes
defined in mesa_frames.abstract. It provides ready-to-use classes for building
agent-based models using DataFrame-based storage, with support Polars backends.

Subpackages:
    polars: Contains Polars-based implementations of agent sets, mixins, and spatial structures.

Modules:
    agents: Defines the AgentsDF class, a collection of AgentSetDFs.
    model: Provides the ModelDF class, the base class for models in mesa-frames.

Classes:
    From polars.agentset:
        AgentSetPolars(AgentSetDF, PolarsMixin): A polars-based implementation of the AgentSet.

    From polars.mixin:
        PolarsMixin(DataFrameMixin): A polars-based implementation of DataFrame operations.

    From polars.space:
        GridPolars(GridDF, PolarsMixin): A polars-based implementation of Grid.


    From agents:
        AgentsDF(AgentContainer): A collection of AgentSetDFs. All agents of the model are stored here.

    From model:
        ModelDF: Base class for models in the mesa-frames library.

Usage:
    Users can import the concrete implementations directly from this package:

    from mesa_frames.concrete import ModelDF, AgentsDF
    # For Polars-based implementations
    from mesa_frames.concrete.polars import AgentSetPolars, GridPolars

    class MyModel(ModelDF):
        def __init__(self):
            super().__init__()
            self.agents.add(AgentSetP(self))
            self.space = GridPolars(self, dimensions=[10, 10])
            # ... other initialization code


For more detailed information on each class, refer to their respective module
and class docstrings.
"""
