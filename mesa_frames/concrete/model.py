from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from mesa_frames.abstract.space import SpaceDF
from mesa_frames.concrete.agents import AgentsDF

if TYPE_CHECKING:
    from mesa_frames.abstract.agents import AgentSetDF


class ModelDF:
    """Base class for models in the mesa-frames library.

    This class serves as a foundational structure for creating agent-based models.
    It includes the basic attributes and methods necessary for initializing and
    running a simulation model.

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

    Properties
    ----------
    agents : AgentsDF
        An AgentSet containing all agents in the model, generated from the _agents attribute.
    agent_types : list of type
        A list of different agent types present in the model.
    """

    random: np.random.Generator
    running: bool
    _seed: int | Sequence[int]
    _agents: AgentsDF  # Where the agents are stored
    _space: SpaceDF | None  # This will be a MultiSpaceDF object

    def __init__(self, seed: int | Sequence[int] | None = None) -> None:
        """Create a new model. Overload this method with the actual code to
        start the model. Always start with super().__init__(seed) to initialize the
        model object properly.

        Parameters
        ----------
        seed : int | Sequence[int] | None, optional
            The seed for the model's generator
        """
        self.random = None
        self.reset_randomizer(seed)
        self.running = True
        self.current_id = 0
        self._agents = AgentsDF(self)
        self._space = None

    def get_agents_of_type(self, agent_type: type) -> "AgentSetDF":
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
            if isinstance(agentset, agent_type):
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
        """A single step. The default method calls the step() method of all agents. Overload as needed."""
        self.agents.step()

    @property
    def agents(self) -> AgentsDF:
        try:
            return self._agents
        except AttributeError:
            raise ValueError(
                "You haven't called super().__init__() in your model. Make sure to call it in your __init__ method."
            )

    @agents.setter
    def agents(self, agents: AgentsDF) -> None:
        if not isinstance(agents, AgentsDF):
            raise TypeError("agents must be an instance of AgentsDF")
        self._agents = agents

    @property
    def agent_types(self) -> list[type]:
        return [agent.__class__ for agent in self._agents._agentsets]

    @property
    def space(self) -> SpaceDF:
        if not self._space:
            raise ValueError(
                "You haven't set the space for the model. Use model.space = your_space"
            )
        return self._space

    @space.setter
    def space(self, space: SpaceDF) -> None:
        self._space = space
