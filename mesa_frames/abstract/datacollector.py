"""
Abstract base classes for data collection components in mesa-frames.

This module defines the core abstractions for data collection in mesa-frames.
It provides a standardized interface for collecting model- and agent-level
data during simulation runs, supporting flexible triggers, custom statistics,
and optional external storage.

Classes:
    AbstractDataCollector:
        An abstract base class defining the structure and core logic for
        all data collector implementations. It supports flexible reporting
        of model and agent attributes, conditional data collection using
        triggers, and pluggable backends for storage.

These classes are designed to be subclassed by concrete implementations that
handle the specifics of data collection and storage such as in-memory, CSV,
or database-backed collectors, potentially using Polars for high-performance
tabular operations.

Usage:
    These classes should not be instantiated directly. Instead, they should be
    subclassed to create concrete DataCollector:

    from mesa_frames.abstract.datacollector import AbstractDataCollector

    class DataCollector(AbstractDataCollector):
        def collect(self):
            # Implementation using Polars DataFrame to collect model and agent data
            ...

        def conditional_collect(self):
            # Implementation using Polars DataFrame to collect model and agent data if trigger returns True
            ...

        def data(self):
            # Returns the data currently in memory
            ...

        def flush(self):
            # Persists collected data if configured and optionally deletes data from memory
            ...

For more detailed information on each class, refer to their individual docstrings.
"""

from abc import ABC, abstractmethod
from typing import Any, Literal
from collections.abc import Callable
from mesa_frames import Model
import polars as pl
import threading
from concurrent.futures import ThreadPoolExecutor


class AbstractDataCollector(ABC):
    """
    Abstract Base Class for Mesa-Frames DataCollector.

    This class defines methods for collecting data from both model and agents.
    Sub classes must implement logic for the methods
    """

    _model: Model
    _model_reporters: dict[str, Callable] | None
    _agent_reporters: dict[str, str | Callable] | None
    _trigger: Callable[..., bool] | None
    _reset_memory = bool
    _storage: Literal["memory", "csv", "parquet", "S3-csv", "S3-parquet", "postgresql"]
    _frames: list[pl.DataFrame]

    def __init__(
        self,
        model: Model,
        model_reporters: dict[str, Callable] | None,
        agent_reporters: dict[str, str | Callable] | None,
        trigger: Callable[[Any], bool] | None,
        reset_memory: bool,
        storage: Literal[
            "memory", "csv", "parquet", "S3-csv", "S3-parquet", "postgresql"
        ],
        max_workers: int,
    ):
        """
        Initialize a Datacollector.

        Parameters
        ----------
        model : Model
            The model object from which data is collected.
        model_reporters : dict[str, Callable] | None
            Functions to collect data at the model level.
        agent_reporters : dict[str, str | Callable] | None
            Agent-level reporters. Values may be:
            - str or list[str]: pull existing columns from each set; columns are suffixed per-set.
            - Callable[[AbstractAgentSetRegistry], Series | DataFrame | dict[str, Series|DataFrame]]: registry-level, runs once per step.
            - Callable[[mesa_frames.abstract.agentset.AbstractAgentSet], Series | DataFrame]: set-level, runs once per set.
            Note: model-level callables are not supported for agent reporters.
        trigger : Callable[[Any], bool] | None
            A function(model) -> bool that determines whether to collect data.
        reset_memory : bool
            Whether to reset in-memory data after flushing. Default is True.
        storage : Literal["memory", "csv", "parquet", "S3-csv", "S3-parquet", "postgresql"        ]
            Storage backend URI (e.g. 'memory:', 'csv:', 'postgresql:').
        max_workers : int
            Maximum number of worker threads used for flushing collected data asynchronously
        """
        self._model = model
        self._model_reporters = model_reporters or {}
        self._agent_reporters = agent_reporters or {}
        self._trigger = trigger or (lambda model: False)
        self._reset_memory = reset_memory
        self._storage = storage or "memory"
        self._frames = []
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def collect(self) -> None:
        """
        Trigger Data collection.

        This method calls _collect() to perform actual data collection.

        Example
        -------
        >>> datacollector.collect()
        """
        self._collect()

    def conditional_collect(self) -> None:
        """
        Trigger data collection if condition is met.

        This method calls _collect() to perform actual data collection only if trigger returns True

        Example
        -------
        >>> datacollector.conditional_collect()
        """
        if self._should_collect():
            self._collect()

    def _should_collect(self) -> bool:
        """
        Evaluate whether data should be collected at current step.

        Returns
        -------
        bool
            True if the configured trigger condition is met, False otherwise.
        """
        return self._trigger(self._model)

    @abstractmethod
    def _collect(self):
        """
        Perform the actual data collection logic.

        This method must be im
        """
        pass

    @property
    @abstractmethod
    def data(self) -> Any:
        """
        Returns collected data currently in memory as a dataframe.

        Example:
        -------
        >>> df = datacollector.data
        >>> print(df)
        """
        pass

    def flush(self) -> None:
        """
        Persist all collected data to configured backend.

        After flushing data optionally clears in-memory
        data buffer if `reset_memory` is True (default behavior).

        use this method to save collected data.


        Example
        -------
        >>> datacollector.flush()
        >>> # Data is saved externally and in-memory buffers are cleared if configured
        """
        with self._lock:
            frames_to_flush = self._frames
            if self._reset_memory:
                self._reset()

        self._executor.submit(self._flush, frames_to_flush)

    def _reset(self):
        """
        Clear all collected data currently stored in memory.

        Use this to free memory or start fresh without affecting persisted data.

        """
        self._frames = []

    @abstractmethod
    def _flush(self) -> None:
        """
        Implement persistence of collected data to external storage.

        This method must be implemented by subclasses to handle
        backend-specific data saving operations.
        """
        pass

    @property
    def seed(self) -> int:
        """
        Function to get the model seed.

        Example:
        --------
        >>> seed = datacollector.seed
        """
        return self._model._seed
