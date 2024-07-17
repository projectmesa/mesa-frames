from abc import abstractmethod
from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING

from numpy.random import Generator
from typing_extensions import Self

from mesa_frames.abstract.agents import AgentContainer
from mesa_frames.abstract.mixin import CopyMixin, DataFrameMixin
from mesa_frames.types_ import (
    DataFrame,
    GeoDataFrame,
    IdsLike,
    SpaceCoordinate,
    SpaceCoordinates,
)

ESPG = int

if TYPE_CHECKING:
    from mesa_frames.concrete.model import ModelDF


class SpaceDF(CopyMixin, DataFrameMixin):
    _model: "ModelDF"
    _agents: DataFrame | GeoDataFrame

    def __init__(self, model: "ModelDF") -> None:
        """Create a new SpaceDF object.

        Parameters
        ----------
        model : 'ModelDF'

        Returns
        -------
        None
        """
        self._model = model

    def random_agents(
        self,
        n: int,
        seed: int | None = None,
    ) -> DataFrame:
        """Return a random sample of agents from the space.

        Parameters
        ----------
        n : int
            The number of agents to sample
        seed : int | None, optional
            The seed for the sampling, by default None
            If None, an integer from the model's random number generator is used.

        Returns
        -------
        DataFrame
            A DataFrame with the sampled agents
        """
        if seed is None:
            seed = self.random.integers(0)
        return self._df_sample(self._agents, n=n, seed=seed)

    @abstractmethod
    def get_directions(
        self,
        pos0: SpaceCoordinate | SpaceCoordinates | None = None,
        pos1: SpaceCoordinate | SpaceCoordinates | None = None,
        agents0: IdsLike | AgentContainer | Collection[AgentContainer] | None = None,
        agents1: IdsLike | AgentContainer | Collection[AgentContainer] | None = None,
    ) -> DataFrame:
        """Returns the directions from pos0 to pos1 or agents0 and agents1.
        If the space is a Network, the direction is the shortest path between the two nodes.
        In all other cases, the direction is the direction vector between the two positions.
        Either positions (pos0, pos1) or agents (agents0, agents1) must be specified, not both and they must have the same length.

        Parameters
        ----------
        pos0 : SpaceCoordinate | SpaceCoordinates | None = None
            The starting positions
        pos1 : SpaceCoordinate | SpaceCoordinates | None = None
            The ending positions
        agents0 : IdsLike | AgentContainer | Collection[AgentContainer] | None = None
            The starting agents
        agents1 : IdsLike | AgentContainer | Collection[AgentContainer] | None = None
            The ending agents

        Returns
        -------
        DataFrame
            A DataFrame where each row represents the direction from pos0 to pos1 or agents0 to agents1
        """
        ...

    @abstractmethod
    def get_distances(
        self,
        pos0: SpaceCoordinate | SpaceCoordinates | None = None,
        pos1: SpaceCoordinate | SpaceCoordinates | None = None,
        agents0: IdsLike | AgentContainer | Collection[AgentContainer] | None = None,
        agents1: IdsLike | AgentContainer | Collection[AgentContainer] | None = None,
    ) -> DataFrame:
        """Returns the distances from pos0 to pos1 or agents0 and agents1.
        If the space is a Network, the distance is the number of nodes of the shortest path between the two nodes.
        In all other cases, the distance is Euclidean/l2/Frobenius norm.
        You should specify either positions (pos0, pos1) or agents (agents0, agents1), not both and they must have the same length.

        Parameters
        ----------
        pos0 : SpaceCoordinate | SpaceCoordinates | None, optional
            The starting positions
        pos1 : SpaceCoordinate | SpaceCoordinates | None, optional
            The ending positions
        agents0 : IdsLike | AgentContainer | Collection[AgentContainer], optional
            The starting agents
        agents1 : IdsLike | AgentContainer | Collection[AgentContainer], optional
            The ending agents

        Returns
        -------
        DataFrame
            A DataFrame where each row represents the distance from pos0 to pos1 or agents0 to agents1
        """
        ...

    @abstractmethod
    def get_neighbors(
        self,
        radius: int | float | Sequence[int] | Sequence[float],
        pos: SpaceCoordinate | SpaceCoordinates | None = None,
        agents: IdsLike | AgentContainer | Collection[AgentContainer] | None = None,
        include_center: bool = False,
    ) -> DataFrame:
        """Get the neighboring agents from given positions or agents according to the specified radiuses.
        Either positions (pos0, pos1) or agents (agents0, agents1) must be specified, not both and they must have the same length.

        Parameters
        ----------
        radius : int | float | Sequence[int] | Sequence[float]
            The radius(es) of the neighborhood
        pos : SpaceCoordinate | SpaceCoordinates | None, optional
            The coordinates of the cell to get the neighborhood from, by default None
        agents : IdsLike | AgentContainer | Collection[AgentContainer] | None, optional
            The id of the agents to get the neighborhood from, by default None
        include_center : bool, optional
            If the center cells or agents should be included in the result, by default False

        Returns
        -------
        DataFrame
            A dataframe with neighboring agents.
            The columns with '_center' suffix represent the center agent/position.

        Raises
        ------
        ValueError
            If both pos and agent are None or if both pos and agent are not None.
        """
        ...

    @abstractmethod
    def move_agents(
        self,
        agents: IdsLike | AgentContainer | Collection[AgentContainer],
        pos: SpaceCoordinate | SpaceCoordinates,
        inplace: bool = True,
    ) -> Self:
        """Place agents in the space according to the specified coordinates. If some agents are already placed,
        raises a RuntimeWarning.

        Parameters
        ----------
        agents : IdsLike | AgentContainer | Collection[AgentContainer]
            The agents to place in the space
        pos : SpaceCoordinate | SpaceCoordinates
            The coordinates for each agents. The length of the coordinates must match the number of agents.
        inplace : bool, optional
            Whether to perform the operation inplace, by default True

        Raises
        ------
        RuntimeWarning
            If some agents are already placed in the space.
        ValueError
            - If some agents are not part of the model.
            - If agents is IdsLike and some agents are present multiple times.

        Returns
        -------
        Self
        """
        ...

    @abstractmethod
    def move_to_empty(
        self,
        agents: IdsLike | AgentContainer | Collection[AgentContainer],
        inplace: bool = True,
    ) -> Self:
        """Move agents to empty cells/positions in the space (cells/positions where there isn't any single agent).

        Parameters
        ----------
        agents : IdsLike | AgentContainer | Collection[AgentContainer]
            The agents to move to empty cells/positions
        inplace : bool, optional
            Whether to perform the operation inplace, by default True

        Returns
        -------
        Self
        """
        ...

    @abstractmethod
    def random_pos(
        self,
        n: int,
        seed: int | None = None,
    ) -> DataFrame:
        """Return a random sample of positions from the space.

        Parameters
        ----------
        n : int
            The number of positions to sample
        seed : int | None, optional
            The seed for the sampling, by default None
            If None, an integer from the model's random number generator is used.

        Returns
        -------
        DataFrame
            A DataFrame with the sampled positions
        """
        ...

    @abstractmethod
    def remove_agents(
        self,
        agents: IdsLike | AgentContainer | Collection[AgentContainer],
        inplace: bool = True,
    ):
        """Remove agents from the space.

        Parameters
        ----------
        agents : IdsLike | AgentContainer | Collection[AgentContainer]
            The agents to remove from the space
        inplace : bool, optional
            Whether to perform the operation inplace, by default True

        Raises
        ------
        ValueError
            If some agents are not part of the model.

        Returns
        -------
        Self
        """
        ...

    @abstractmethod
    def swap_agents(
        self,
        agents0: IdsLike | AgentContainer | Collection[AgentContainer],
        agents1: IdsLike | AgentContainer | Collection[AgentContainer],
    ) -> Self:
        """Swap the positions of the agents in the space.
        agents0 and agents1 must have the same length and all agents must be placed in the space.

        Parameters
        ----------
        agents0 : IdsLike | AgentContainer | Collection[AgentContainer]
            The first set of agents to swap
        agents1 : IdsLike | AgentContainer | Collection[AgentContainer]
            The second set of agents to swap

        Returns
        -------
        Self
        """

    @abstractmethod
    def __repr__(self) -> str: ...

    @abstractmethod
    def __str__(self) -> str: ...

    @property
    def agents(self) -> DataFrame | GeoDataFrame:
        """Get the ids of the agents placed in the cell set, along with their coordinates or geometries

        Returns
        -------
        AgentsDF
        """
        return self._agents

    @property
    def model(self) -> "ModelDF":
        """The model to which the space belongs.

        Returns
        -------
        'ModelDF'
        """
        return self._model

    @property
    def random(self) -> Generator:
        """The model's random number generator.

        Returns
        -------
        Generator
        """
        return self.model.random
