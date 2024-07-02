from collections.abc import Sequence
from typing import Any

import numpy as np

from mesa_frames.abstract.agents import AgentSetDF
from mesa_frames.concrete.agents import AgentsDF


class ModelDF:
    """Base class for models in the mesa-frames library.

    This class serves as a foundational structure for creating agent-based models.
    It includes the basic attributes and methods necessary for initializing and
    running a simulation model.

    Attributes
    ----------
    running : bool
        A boolean indicating if the model should continue running.
    schedule : Any
        An object to manage the order and execution of agent steps.
    current_id : int
        A counter for assigning unique IDs to agents.
    _agents : AgentsDF
        A mapping of each agent type to a dict of its instances.

    Properties
    ----------
    agents : AgentsDF
        An AgentSet containing all agents in the model, generated from the _agents attribute.
    agent_types : list of type
        A list of different agent types present in the model.

    Methods
    -------
    __new__(cls, seed: int | Sequence[int] | None = None, *args: Any, **kwargs: Any) -> Any
        Create a new model object and instantiate its RNG automatically.
    __init__(self, *args: Any, **kwargs: Any) -> None
        Create a new model. Overload this method with the actual code to start the model.
    get_agents_of_type(self, agent_type: type) -> AgentSetDF
        Retrieve the AgentSetDF of a specified type.
    initialize_data_collector(self, model_reporters: dict | None = None, agent_reporters: dict | None = None, tables: dict | None = None) -> None
        Initialize the data collector for the model (not implemented yet).
    next_id(self) -> int
        Generate and return the next unique identifier for an agent (not implemented yet).
    reset_randomizer(self, seed: int | Sequence[int] | None) -> None
        Reset the model random number generator with a new or existing seed.
    run_model(self) -> None
        Run the model until the end condition is reached.
    step(self) -> None
        Execute a single step of the model's simulation process (needs to be overridden in a subclass).
    """

    random: np.random.Generator
    _seed: int | Sequence[int]
    running: bool
    _agents: AgentsDF

    def __new__(
        cls, seed: int | Sequence[int] | None = None, *args: Any, **kwargs: Any
    ) -> Any:
        """Create a new model object and instantiate its RNG automatically."""
        obj = object.__new__(cls)
        obj.reset_randomizer(seed)
        return obj

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Create a new model. Overload this method with the actual code to
        start the model. Always start with super().__init__() to initialize the
        model object properly.
        """
        self.running = True
        self.schedule = None
        self.current_id = 0
        self._agents = AgentsDF()

    def get_agents_of_type(self, agent_type: type) -> AgentSetDF:
        """Retrieve the AgentSetDF of a specified type.

        Parameters
        ----------
        agent_type : type
            The type of AgentSetDF to retrieve.

        Returns
        -------
        AgentSetDF
            The AgentSetDF of the specified type.
        """
        for agentset in self._agents._agentsets:
            if agent_type == type(agentset):
                return agentset
        raise ValueError(f"No agents of type {agent_type} found in the model.")

    def initialize_data_collector(
        self,
        model_reporters: dict | None = None,
        agent_reporters: dict | None = None,
        tables: dict | None = None,
    ) -> None:
        raise NotImplementedError(
            "initialize_data_collector() method not implemented yet for ModelDF"
        )

    def next_id(self) -> int:
        raise NotImplementedError("next_id() method not implemented for ModelDF")

    def reset_randomizer(self, seed: int | Sequence[int] | None) -> None:
        """Reset the model random number generator.

        Parameters:
        ----------
        seed : int | None
            A new seed for the RNG; if None, reset using the current seed
        """
        if seed is None:
            seed = np.random.SeedSequence().entropy
        assert seed is not None
        self._seed = seed
        self.random = np.random.default_rng(seed=self._seed)

    def run_model(self) -> None:
        """Run the model until the end condition is reached. Overload as
        needed.
        """
        while self.running:
            self.step()

    def step(self) -> None:
        """A single step. Fill in here."""
        raise NotImplementedError("step() method needs to be overridden in a subclass.")

    @property
    def agents(self) -> AgentsDF:
        return self._agents

    @agents.setter
    def agents(self, agents: AgentsDF) -> None:
        if not isinstance(agents, AgentsDF):
            raise TypeError("agents must be an instance of AgentsDF")
        self._agents = agents

    @property
    def agent_types(self) -> list[type]:
        return [agent.__class__ for agent in self._agents._agentsets]
