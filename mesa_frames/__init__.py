"""
mesa-frames: High-performance extension for the Mesa agent-based modeling framework.

mesa-frames extends the Mesa framework to support complex simulations with thousands
of agents by storing agents in DataFrames. This approach significantly enhances
performance and scalability while maintaining a syntax similar to Mesa.

Key Features:
- Utilizes DataFrame storage for agents, enabling vectorized operations
- Supports Polars as backend libraries
- Provides similar syntax to Mesa for ease of transition
- Allows for vectorized functions when simultaneous activation of agents is possible
- Implements SIMD processing for optimized simultaneous operations
- Includes Grid for efficient grid-based spatial modeling

Main Components:
- AgentSet: Agent set implementation using Polars backend
- Model: Base model class for mesa-frames
- Grid: Grid space implementation for spatial modeling

Usage:
To use mesa-frames, import the necessary components and subclass them as needed:

    from mesa_frames import AgentSet, Model, Grid

    class MyAgent(AgentSet):
        # Define your agent logic here

    class MyModel(Model):
        def __init__(self, width, height):
            super().__init__()
            self.grid = Grid(self, [width, height])
            # Define your model logic here

Note: mesa-frames is in early development. API and usage patterns may change.

For more detailed information, refer to the full documentation and API reference.

Developed by: Project Mesa, Adam Amer
License: MIT
GitHub: https://github.com/projectmesa/mesa-frames
"""

from __future__ import annotations

import os

# Enable runtime type checking if requested via environment variable
if os.getenv("MESA_FRAMES_RUNTIME_TYPECHECKING", "").lower() in ("1", "true", "yes"):
    try:
        from beartype.claw import beartype_this_package

        beartype_this_package()
    except ImportError:
        import warnings

        warnings.warn(
            "MESA_FRAMES_RUNTIME_TYPECHECKING is enabled but beartype is not installed.",
            ImportWarning,
            stacklevel=2,
        )

from mesa_frames.concrete.agentset import AgentSet
from mesa_frames.concrete.agentsetregistry import AgentSetRegistry
from mesa_frames.concrete.datacollector import DataCollector
from mesa_frames.concrete.model import Model
from mesa_frames.concrete.space import Grid

__all__ = ["AgentSetRegistry", "AgentSet", "Model", "Grid", "DataCollector"]

__version__ = "0.1.1.dev0"
