from collections.abc import Collection
from typing import Literal

from collections.abc import Sequence

import geopandas as gpd
import geopolars as gpl
import pandas as pd
import polars as pl
from numpy import ndarray

####----- Agnostic Types -----####
AgnosticMask = Literal["all", "active"] | None
AgnosticIds = int | Collection[int]

###----- Pandas Types -----###

ArrayLike = pd.api.extensions.ExtensionArray | ndarray
AnyArrayLike = ArrayLike | pd.Index | pd.Series
PandasMaskLike = AgnosticMask | pd.Series | pd.DataFrame | AnyArrayLike
PandasIdsLike = AgnosticIds | pd.Series | pd.Index
PandasGridCapacity = ndarray

###----- Polars Types -----###

PolarsMaskLike = AgnosticMask | pl.Expr | pl.Series | pl.DataFrame | Collection[int]
PolarsIdsLike = AgnosticIds | pl.Series
PolarsGridCapacity = list[pl.Expr]

###----- Generic -----###
GeoDataFrame = gpd.GeoDataFrame | gpl.GeoDataFrame
DataFrame = pd.DataFrame | pl.DataFrame
Series = pd.Series | pl.Series
Series = pd.Series | pl.Series
Index = pd.Index | pl.Series
BoolSeries = pd.Series | pl.Series
MaskLike = AgnosticMask | PandasMaskLike | PolarsMaskLike
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
