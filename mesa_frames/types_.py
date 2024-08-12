from collections.abc import Collection, Sequence
from typing import Literal

import geopandas as gpd
import geopolars as gpl
import pandas as pd
import polars as pl
from numpy import ndarray
from typing_extensions import Any

####----- Agnostic Types -----####
AgnosticMask = (
    Any | Sequence[Any] | None
)  # Any is a placeholder for any type if it's a single value
AgnosticAgentMask = Sequence[int] | int | Literal["all", "active"] | None
AgnosticIds = int | Collection[int]

###----- Pandas Types -----###

PandasMask = pd.Series | pd.DataFrame | AgnosticMask
AgentPandasMask = AgnosticAgentMask | pd.Series | pd.DataFrame
PandasIdsLike = AgnosticIds | pd.Series | pd.Index
PandasGridCapacity = ndarray

###----- Polars Types -----###

PolarsMask = pl.Expr | pl.Series | pl.DataFrame | AgnosticMask
AgentPolarsMask = AgnosticAgentMask | pl.Expr | pl.Series | pl.DataFrame | Sequence[int]
PolarsIdsLike = AgnosticIds | pl.Series
PolarsGridCapacity = list[pl.Expr]

###----- Generic -----###
GeoDataFrame = gpd.GeoDataFrame | gpl.GeoDataFrame
DataFrame = pd.DataFrame | pl.DataFrame
Series = pd.Series | pl.Series
Index = pd.Index | pl.Series
BoolSeries = pd.Series | pl.Series
Mask = PandasMask | PolarsMask
AgentMask = AgentPandasMask | AgentPolarsMask
IdsLike = AgnosticIds | PandasIdsLike | PolarsIdsLike


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

GridCapacity = PandasGridCapacity | PolarsGridCapacity
NetworkCapacity = DataFrame

DiscreteSpaceCapacity = GridCapacity | NetworkCapacity
