"""
Polars-based implementations for mesa-frames.

This subpackage contains concrete implementations of mesa-frames components
using Polars as the backend for DataFrame operations. It provides high-performance,
Polars-based classes for agent sets, spatial structures, and DataFrame operations.

Modules:
    agentset: Defines the AgentSetPolars class, a Polars-based implementation of AgentSet.
    mixin: Provides the PolarsMixin class, implementing DataFrame operations using Polars.
    space: Contains the GridPolars class, a Polars-based implementation of Grid.

Classes:
    AgentSetPolars(AgentSetDF, PolarsMixin):
        A Polars-based implementation of the AgentSet, using Polars DataFrames
        for efficient agent storage and manipulation.

    PolarsMixin(DataFrameMixin):
        A mixin class that implements DataFrame operations using Polars,
        providing methods for data manipulation and analysis.

    GridPolars(GridDF, PolarsMixin):
        A Polars-based implementation of Grid, using Polars DataFrames for
        efficient spatial operations and agent positioning.

Usage:
    These classes can be imported and used directly in mesa-frames models:

    from mesa_frames.concrete.polars import AgentSetPolars, GridPolars
    from mesa_frames.concrete.model import ModelDF

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
