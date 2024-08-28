"""
Pandas-based implementations for mesa-frames.

This subpackage contains concrete implementations of mesa-frames components
using pandas as the backend for DataFrame operations. It provides high-performance,
pandas-based classes for agent sets, spatial structures, and DataFrame operations.

Modules:
    agentset: Defines the AgentSetPandas class, a pandas-based implementation of AgentSet.
    mixin: Provides the PandasMixin class, implementing DataFrame operations using pandas.
    space: Contains the GridPandas class, a pandas-based implementation of Grid.

Classes:
    AgentSetPandas(AgentSetDF, PandasMixin):
        A pandas-based implementation of the AgentSet, using pandas DataFrames
        for efficient agent storage and manipulation.

    PandasMixin(DataFrameMixin):
        A mixin class that implements DataFrame operations using pandas,
        providing methods for data manipulation and analysis.

    GridPandas(GridDF, PandasMixin):
        A pandas-based implementation of Grid, using pandas DataFrames for
        efficient spatial operations and agent positioning.

Usage:
    These classes can be imported and used directly in mesa-frames models:

    from mesa_frames.concrete.pandas import AgentSetPandas, GridPandas
    from mesa_frames.concrete.model import ModelDF

    class MyAgents(AgentSetPandas):
        def __init__(self, model):
            super().__init__(model)
            # Initialize agents

    class MyModel(ModelDF):
        def __init__(self, width, height):
            super().__init__()
            self.agents.add(MyAgents(self))
            self.space = GridPandas(self, dimensions=[width, height])

Note:
    Using these pandas-based implementations requires pandas to be installed.
    The performance characteristics will depend on the pandas version and the
    specific operations used in the model.

For more detailed information on each class, refer to their respective module
and class docstrings.
"""
