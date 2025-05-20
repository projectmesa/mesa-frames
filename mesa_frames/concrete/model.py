"""
Concrete implementation of the model class for mesa-frames.

This module provides the concrete implementation of the base model class for
the mesa-frames library. It defines the ModelDF class, which serves as the
foundation for creating agent-based models using DataFrame-based agent storage.

Classes:
    ModelDF:
        The base class for models in the mesa-frames library. This class
        provides the core functionality for initializing and running
        agent-based simulations using DataFrame-backed agent sets.

The ModelDF class is designed to be subclassed by users to create specific
model implementations. It provides the basic structure and methods necessary
for setting up and running simulations, while leveraging the performance
benefits of DataFrame-based agent storage.

Usage:
    To create a custom model, subclass ModelDF and implement the necessary
    methods:

    from mesa_frames.concrete.model import ModelDF
    from mesa_frames.concrete.agents import AgentSetPolars

    class MyCustomModel(ModelDF):
        def __init__(self, num_agents):
            super().__init__()
            self.agents += AgentSetPolars(self)
            # Initialize your model-specific attributes and agent sets

        def run_model(self):
            # Implement the logic for a single step of your model
            for _ in range(10):
                self.step()

        # Add any other custom methods for your model

For more detailed information on the ModelDF class and its methods, refer to
the class docstring.
"""

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

    """

    random: np.random.Generator
    running: bool
    _seed: int | Sequence[int]
    _agents: AgentsDF  # Where the agents are stored
    _space: SpaceDF | None  # This will be a MultiSpaceDF object

    def __init__(self, seed: int | Sequence[int] | None = None) -> None:
        """Create a new model.

        Overload this method with the actual code to
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
        self._steps = 0

        self._user_step = self.step
        self.step = self._wrapped_step

    def _wrapped_step(self) -> None:
        """Automatically increments step counter and calls user-defined step()."""
        self._steps += 1
        self._user_step()

    @property
    def steps(self) -> int:
        """Get the current step count."""
        return self._steps

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

    def reset_randomizer(self, seed: int | Sequence[int] | None) -> None:
        """Reset the model random number generator.

        Parameters
        ----------
        seed : int | Sequence[int] | None
            A new seed for the RNG; if None, reset using the current seed
        """
        if seed is None:
            seed = np.random.SeedSequence().entropy
        assert seed is not None
        self._seed = seed
        self.random = np.random.default_rng(seed=self._seed)

    def run_model(self) -> None:
        """Run the model until the end condition is reached.

        Overload as needed.
        """
        while self.running:
            self.step()

    def step(self) -> None:
        """Run a single step.

        The default method calls the step() method of all agents. Overload as needed.
        """
        self.agents.step()

    @property
    def steps(self) -> int:
        """Get the current step count.

        Returns
        -------
        int
            The current step count of the model.
        """
        return self._steps

    @property
    def agents(self) -> AgentsDF:
        """Get the AgentsDF object containing all agents in the model.

        Returns
        -------
        AgentsDF
            The AgentsDF object containing all agents in the model.

        Raises
        ------
        ValueError
            If the model has not been initialized properly with super().__init__().
        """
        try:
            return self._agents
        except AttributeError:
            if __debug__:  # Only execute in non-optimized mode
                raise RuntimeError(
                    "You haven't called super().__init__() in your model. Make sure to call it in your __init__ method."
                )

    @agents.setter
    def agents(self, agents: AgentsDF) -> None:
        if __debug__:  # Only execute in non-optimized mode
            if not isinstance(agents, AgentsDF):
                raise TypeError("agents must be an instance of AgentsDF")

        self._agents = agents

    @property
    def agent_types(self) -> list[type]:
        """Get a list of different agent types present in the model.

        Returns
        -------
        list[type]
            A list of the different agent types present in the model.
        """
        return [agent.__class__ for agent in self._agents._agentsets]

    @property
    def space(self) -> SpaceDF:
        """Get the space object associated with the model.

        Returns
        -------
        SpaceDF
            The space object associated with the model.

        Raises
        ------
        ValueError
            If the space has not been set for the model.
        """
        if not self._space:
            raise ValueError(
                "You haven't set the space for the model. Use model.space = your_space"
            )
        return self._space

    @space.setter
    def space(self, space: SpaceDF) -> None:
        self._space = space
