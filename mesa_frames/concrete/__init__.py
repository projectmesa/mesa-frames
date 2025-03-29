"""
Concrete implementations of mesa-frames components.

This package provides concrete implementations of the abstract base
classes defined in mesa_frames.abstract. It offers ready-to-use
components for building agent-based models with a DataFrame-based storage system.

The implementation leverages Polars as the backend for high-performance DataFrame operations.
It includes optimized classes for agent sets, spatial structures, and data manipulation,
ensuring efficient model execution.

Subpackages:
    polars: Contains Polars-based implementations of agent sets, mixins, and spatial structures.

Modules:
    agents: Defines the AgentsDF class, a collection of AgentSetDFs.
    model: Provides the ModelDF class, the base class for models in mesa-frames.
    agentset: Defines the AgentSetPolars class, a Polars-based implementation of AgentSet.
    mixin: Provides the PolarsMixin class, implementing DataFrame operations using Polars.
    space: Contains the GridPolars class, a Polars-based implementation of Grid.

Classes:
    from agentset:
        AgentSetPolars(AgentSetDF, PolarsMixin):
            A Polars-based implementation of the AgentSet, using Polars DataFrames
            for efficient agent storage and manipulation.

    from mixin:
        PolarsMixin(DataFrameMixin):
            A mixin class that implements DataFrame operations using Polars,
            providing methods for data manipulation and analysis.
    from space:
        GridPolars(GridDF, PolarsMixin):
            A Polars-based implementation of Grid, using Polars DataFrames for
            efficient spatial operations and agent positioning.

    From agents:
        AgentsDF(AgentContainer): A collection of AgentSetDFs. All agents of the model are stored here.

    From model:
        ModelDF: Base class for models in the mesa-frames library.

Usage:
    Users can import the concrete implementations directly from this package:

    from mesa_frames.concrete import ModelDF, AgentsDF
    # For Polars-based implementations
    from mesa_frames.concrete import AgentSetPolars, GridPolars
    from mesa_frames.concrete.model import ModelDF

    class MyModel(ModelDF):
        def __init__(self):
            super().__init__()
            self.agents.add(AgentSetPolars(self))
            self.space = GridPolars(self, dimensions=[10, 10])
            # ... other initialization code

        from mesa_frames.concrete import AgentSetPolars, GridPolars

    class MyAgents(AgentSetPolars):
        def __init__(self, model):
            super().__init__(model)
            # Initialize agents

    class MyModel(ModelDF):
        def __init__(self, width, height):
            super().__init__()
            self.agents = MyAgents(self)
            self.grid = GridPolars(width, height, self)
Features:
    - High-performance DataFrame operations using Polars
    - Efficient memory usage and fast computation
    - Support for lazy evaluation and query optimization
    - Seamless integration with other mesa-frames components

Note:
    Using these Polars-based implementations requires Polars to be installed.
    Polars offers excellent performance for large datasets and complex operations,
    making it suitable for large-scale agent-based models.


For more detailed information on each class, refer to their respective module
and class docstrings.
"""
