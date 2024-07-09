"""
Mesa Frames Space Module
=================

Objects used to add a spatial component to a model.

Grid: base grid, which creates a rectangular grid.
SingleGrid: extension to Grid which strictly enforces one agent per cell.
MultiGrid: extension to Grid where each cell can contain a set of agents.
HexGrid: extension to Grid to handle hexagonal neighbors.
ContinuousSpace: a two-dimensional space where each agent has an arbitrary
                 position of `float`'s.
NetworkGrid: a network where each node contains zero or more agents.
"""

from mesa_frames.abstract.agents import AgentContainer
from mesa_frames.types_ import IdsLike, PositionsLike


class SpaceDF:
    def _check_empty_pos(pos: PositionsLike) -> bool:
        """Check if the given positions are empty.

        Parameters
        ----------
        pos : DataFrame | tuple[Series, Series] | Series
            Input positions to check.

        Returns
        -------
        Series[bool]
            Whether
        """


class SingleGrid(SpaceDF):
    """Rectangular grid where each cell contains exactly at most one agent.

    Grid cells are indexed by [x, y], where [0, 0] is assumed to be the
    bottom-left and [width-1, height-1] is the top-right. If a grid is
    toroidal, the top and bottom, and left and right, edges wrap to each other.

    This class provides a property `empties` that returns a set of coordinates
    for all empty cells in the grid. It is automatically updated whenever
    agents are added or removed from the grid. The `empties` property should be
    used for efficient access to current empty cells rather than manually
    iterating over the grid to check for emptiness.

    """

    def place_agents(self, agents: IdsLike | AgentContainer, pos: PositionsLike):
        """Place agents on the grid at the coordinates specified in pos.
        NOTE: The cells must be empty.


        Parameters
        ----------
        agents : IdsLike | AgentContainer

        pos : DataFrame | tuple[Series, Series]
            _description_
        """

    def _check_empty_pos(pos: PositionsLike) -> bool:
        """Check if the given positions are empty.

        Parameters
        ----------
        pos : DataFrame | tuple[Series, Series]
            Input positions to check.

        Returns
        -------
        bool
            _description_
        """
