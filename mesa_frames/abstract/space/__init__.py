"""Abstract space interfaces."""

from .cells import AbstractCells
from .neighborhood import AbstractNeighborhood
from .discrete import AbstractDiscreteSpace
from .grid import AbstractGrid
from .space import Space

__all__ = [
    "AbstractCells",
    "AbstractNeighborhood",
    "AbstractDiscreteSpace",
    "AbstractGrid",
    "Space",
]
