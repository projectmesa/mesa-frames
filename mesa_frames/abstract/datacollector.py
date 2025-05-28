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
or database-backed collectors, potentially using Polars forhigh-performance 
tabular operations.

Usage:
    These classes should not be instantiated directly. Instead, they should be
    subclassed to create concrete datacolllector:

    from mesa_frames.abstract.datacolllector import AbstractDataCollector

    class DataCollector(AbstractDataCollector):
        def _collect(self):
            # Implementation using Polars DataFrame to collect model and agent data
            ...

        def get_data(self):
            # Returns the collected data
            ...

        def flush(self):
            # Persists collected data if configured
            ...

        def register_stat(self, name, func):
            # Registers a custom statistic for post-processing
            ...
The `collect()` method in the abstract base class automatically evaluates
the trigger condition before invoking `_collect()`. Subclasses must implement
`_collect()` along with `get_data()`, `flush()`, and `register_stat()`.

For more detailed information on each class, refer to their individual docstrings.
"""
from abc import ABC,abstractmethod
from typing import Callable, Dict, Optional, Union, Any
from agents import ModelDF

class AbstractDataCollector(ABC):

    _model: ModelDF

    def __init__(
        self,
        model: Any,
        model_reporters: Optional[Dict[str, Union[str, Callable]]] = None,
        agent_reporters: Optional[Dict[str, Union[str, Callable]]] = None,
        trigger: Optional[Callable[[Any], bool]] = None,
        storage: Optional[str] = None
    ):
        self._model = model 
        self._model_reporters = model_reporters or {}
        self._agent_reporters = agent_reporters or {}
        self._trigger = trigger or (lambda model: True)
        self._storage_uri = storage or "memory:"
        self._frames = [] 

    def collect(self) -> None:
        """ collect data - doesn't materialise until called"""
        if self.should_collect():
            self._collect() 

    @abstractmethod
    def _collect(self):
        """ actual collection logic"""
        pass

    def should_collect(self) -> bool:
        """trigger == True"""
        return self._trigger(self._model)

    @abstractmethod
    def get_data(self) -> Any:
        """ returns collected data currently in memory"""
        pass

    @abstractmethod
    def _flush(self) -> None:
        """if configured persists to external"""
        pass
