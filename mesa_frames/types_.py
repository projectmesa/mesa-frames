"""Type aliases for the mesa_frames package."""

from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING, Literal, Union

if TYPE_CHECKING:
    from mesa_frames import AgentSetPolars


# import geopolars as gpl
import polars as pl
from numpy import ndarray
from typing_extensions import Any

####----- Agnostic Types -----####
AgnosticMask = (
    Any | Sequence[Any] | None
)  # Any is a placeholder for any type if it's a single value
AgnosticAgentMask = Sequence[int] | int | Literal["all", "active"] | None
AgnosticIds = int | Collection[int]

###----- Polars Types -----###

PolarsMask = pl.Expr | pl.Series | pl.DataFrame | AgnosticMask
AgentPolarsMask = AgnosticAgentMask | pl.Expr | pl.Series | pl.DataFrame | Sequence[int]
PolarsIdsLike = AgnosticIds | pl.Series
PolarsGridCapacity = list[pl.Expr]

###----- Generic -----###
# GeoDataFrame = gpd.GeoDataFrame | gpl.GeoDataFrame
DataFrame = pl.DataFrame
DataFrameInput = dict[str, Any] | Sequence[Sequence] | DataFrame
Series = pl.Series
Index = pl.Series
BoolSeries = pl.Series
Mask = PolarsMask
AgentMask = AgentPolarsMask
IdsLike = AgnosticIds | PolarsIdsLike
ArrayLike = ndarray | Series | Sequence

###----- Time ------###
TimeT = float | int

###----- Space -----###

NetworkCoordinate = int | DataFrame

GridCoordinate = int | Sequence[int] | DataFrame

DiscreteCoordinate = NetworkCoordinate | GridCoordinate
ContinousCoordinate = float | Sequence[float] | DataFrame

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

GridCapacity = PolarsGridCapacity
NetworkCapacity = DataFrame

DiscreteSpaceCapacity = GridCapacity | NetworkCapacity
