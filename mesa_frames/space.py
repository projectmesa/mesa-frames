from mesa_frames.base.agent import AgentSetPandas
from mesa.space import MultiGrid
from numpy import shape
:
from numpy.typing import ArrayLike
from numpy import ndarray
from mesa.space import _Grid

class _GridDF(_Grid):
    def __init__(self, width: int, height: int, torus: bool) -> None:
        """Create a new grid.

        Args:
            width, height: The width and height of the grid
            torus: Boolean whether the grid wraps or not.
        """
        self.height = height
        self.width = width
        self.torus = torus
        self.num_cells = height * width

        # Internal numpy array which stores the index of the agents at each cell.
        self._grid: ndarray = ndarray((width, height), dtype=int)

        # Flag to check if the empties set has been created. Better than initializing
        # _empties as set() because in this case it would become impossible to discern
        # if the set hasn't still being built or if it has become empty after creation.
        self._empties_built = False

        # Neighborhood Cache
        self._neighborhood_cache: dict[Any, Sequence[Coordinate]] = {}

        # Cutoff used inside self.move_to_empty. The parameters are fitted on Python
        # 3.11 and it was verified that they are roughly the same for 3.10. Refer to
        # the code in PR#1565 to check for their stability when a new release gets out.
        self.cutoff_empties = 7.953 * self.num_cells**0.384


class MultiGridDF(MultiGrid):
    def place_agent(self, agents: AgentSetPandas, pos: ArrayLike) -> None:
        """Place the agents at the specified location, and set its pos variable.
        
        Parameters
        ----------
        agents : Agent
            The agent object to place.
        
        pos : ArrayLike
            Array of x, y pos values
        """
        if shape(pos) != (len(agents), 2):
            raise ValueError("Pos must be an array of two values")
        
        if agent.pos is None or agent not in self._grid[x][y]:
            self._grid[x][y].append(agent)
            agent.pos = pos
            if self._empties_built:
                self._empties.discard(pos)
                self._empty_mask[agent.pos] = True
