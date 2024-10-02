"""Type aliases for the mesa_frames package."""

from collections.abc import Collection, Sequence
from typing import Literal

import ibis as ib
from numpy import ndarray
from typing_extensions import Any

###----- Generic -----###
DataFrameInput = dict[str, Any] | Sequence[Sequence] | ib.Table
BoolColumn = ib.ir.BooleanColumn
Mask = ib.Expr
AgentMask = Sequence[int] | int | Literal["all", "active"] | ib.Column | ib.Table | None
IdsLike = int | Collection[int] | ib.ir.IntegerColumn | ib.Table
ArrayLike = ndarray | ib.Column | Sequence

###----- Time ------###
TimeT = float | int

###----- Space -----###

NetworkCoordinate = int | ib.Table

GridCoordinate = int | Sequence[int] | ib.Table

DiscreteCoordinate = NetworkCoordinate | GridCoordinate
ContinousCoordinate = float | Sequence[float] | ib.Table

SpaceCoordinate = DiscreteCoordinate | ContinousCoordinate


NetworkCoordinates = NetworkCoordinate | Collection[NetworkCoordinate]
GridCoordinates = (
    GridCoordinate | Sequence[int | slice | Sequence[int]] | Collection[GridCoordinate]
)

DiscreteCoordinates = NetworkCoordinates | GridCoordinates
ContinousCoordinates = (
    ContinousCoordinate
    | Sequence[float | Sequence[float]]
    | Collection[ContinousCoordinate]
)

SpaceCoordinates = DiscreteCoordinates | ContinousCoordinates

GridCapacity = ndarray
NetworkCapacity = ib.Table

DiscreteSpaceCapacity = GridCapacity | NetworkCapacity
