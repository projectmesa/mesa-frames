"""
mesa-frames: High-performance extension for the Mesa agent-based modeling framework.

mesa-frames extends the Mesa framework to support complex simulations with thousands
of agents by storing agents in DataFrames. This approach significantly enhances
performance and scalability while maintaining a syntax similar to Mesa.

Key Features:
- Utilizes DataFrame storage for agents, enabling vectorized operations
- Supports both pandas and Polars as backend libraries
- Provides similar syntax to Mesa for ease of transition
- Allows for vectorized functions when simultaneous activation of agents is possible
- Implements SIMD processing for optimized simultaneous operations
- Includes GridDF for efficient grid-based spatial modeling

Main Components:
- AgentSetPandas: Agent set implementation using pandas backend
- AgentSetPolars: Agent set implementation using Polars backend
- ModelDF: Base model class for mesa-frames
- GridDF: Grid space implementation for spatial modeling

Usage:
To use mesa-frames, import the necessary components and subclass them as needed:

    from mesa_frames import AgentSetPolars, ModelDF, GridDF

    class MyAgent(AgentSetPolars):
        # Define your agent logic here

    class MyModel(ModelDF):
        def __init__(self, width, height):
            super().__init__()
            self.grid = GridDF(width, height, self)
            # Define your model logic here

Note: mesa-frames is in early development. API and usage patterns may change.

For more detailed information, refer to the full documentation and API reference.

Developed by: Adam Amer
License: MIT
GitHub: https://github.com/adamamer20/mesa_frames
"""

from mesa_frames.concrete.agents import AgentsDF
from mesa_frames.concrete.model import ModelDF
from mesa_frames.concrete.pandas.agentset import AgentSetPandas
from mesa_frames.concrete.pandas.space import GridPandas
from mesa_frames.concrete.polars.agentset import AgentSetPolars
from mesa_frames.concrete.polars.space import GridPolars

__all__ = [
    "AgentsDF",
    "AgentSetPandas",
    "AgentSetPolars",
    "ModelDF",
    "GridPandas",
    "GridPolars",
]
