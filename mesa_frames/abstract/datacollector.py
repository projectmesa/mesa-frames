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

        def data(self):
            # Returns the data currently in memory
            ...

        def flush(self):
            # Persists collected data if configured and optionally deletes data from memory
            ...

The `collect()` method in the abstract base class automatically evaluates
the trigger condition before invoking `_collect()`. Subclasses must implement
`_collect()` along with `data()`, `flush()`.

For more detailed information on each class, refer to their individual docstrings.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Union, Any
from collections.abc import Callable
from agents import ModelDF

class AbstractDataCollector(ABC):
    """
    Abstract Base Class for Mesa-Frames DataCollector.

    This class defines methods for collecting data from both model and agents.
    Sub classes must implement logic for the methods
    """

    _model: ModelDF
    _model_reporters: Optional[Dict[str, Callable]]
    _agent_reporters: Optional[Dict[str, Union[str, Callable]]]
    _trigger: Callable[...,bool]
    _reset_memory = bool
    _storage_uri: Literal["memory:", "csv:", "postgresql:"]
    _frames: List[pl.DataFrame]

    def __init__(
        self,
        model: ModelDF,
        model_reporters: Optional[Dict[str, Callable]] = None,
        agent_reporters: Optional[Dict[str, Union[str, Callable]]] = None,
        trigger: Optional[Callable[[Any], bool]] = None,
        reset_memory : bool = True,
        storage: Optional[str] = None
    ):
        """
        Initialize a Datacollector

        Parameters
        ----------
        model : ModelDF
            The model object from which data is collected.
        model_reporters : dict[str, Callable], optional
            Functions to collect data at the model level.
        agent_reporters : dict[str, Union[str, Callable]], optional
            Attributes or functions to collect data at the agent level.
        trigger : Callable, optional
            A function(model) -> bool that determines whether to collect data.
        reset_memory : bool
            Whether to reset in-memory data after flushing. Default is True.
        storage : str
            Storage backend URI (e.g. 'memory:', 'csv:', 'postgresql:').
        """
        self._model = model
        self._model_reporters = model_reporters or {}
        self._agent_reporters = agent_reporters or {}
        self._trigger = trigger or (lambda model: True)
        self._reset_memory = reset_memory
        self._storage_uri = storage or "memory:"
        self._frames = []

    def collect(self) -> None:
        """ Trigger data collection if condition is met"""
        if self.should_collect():
            self._collect() 

    def should_collect(self) -> bool:
        """ Evaluates whether data should be collected or not"""
        return self._trigger(self._model)

    @abstractmethod
    def _collect(self):
        """ performs actual data collection"""
        pass
    
    @property
    @abstractmethod
    def get_data(self) -> Any:
        """ returns collected data currently in memory as a dataframe"""
        pass
    
    #def load_data(self,step:Optional[int]=None):

    def flush(self) ->None:
        """public method to flush the collected data and optionally reset in-memory data """
        self._flush()
        if self._reset_memory:
            self.reset()
    
    def reset(self):
        """ method to reset the data in memory"""
        self._frames = []

    @abstractmethod
    def _flush(self) -> None:
        """Persists collected data to external storage and"""
        pass
