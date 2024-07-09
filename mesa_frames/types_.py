from collections.abc import Collection

import geopandas as gpd
import geopolars as gpl
import pandas as pd
import polars as pl
from numpy import ndarray
from typing_extensions import Literal, Sequence

####----- Agnostic Types -----####
AgnosticMask = Literal["all", "active"] | None
AgnosticIds = int | Collection[int]

###----- Pandas Types -----###

ArrayLike = pd.api.extensions.ExtensionArray | ndarray
AnyArrayLike = ArrayLike | pd.Index | pd.Series
PandasMaskLike = AgnosticMask | pd.Series | pd.DataFrame | AnyArrayLike
PandasIdsLike = AgnosticIds | pd.Series | pd.Index

###----- Polars Types -----###

PolarsMaskLike = AgnosticMask | pl.Expr | pl.Series | pl.DataFrame | Collection[int]
PolarsIdsLike = AgnosticIds | pl.Series

###----- Generic -----###

GeoDataFame = gpd.GeoDataFrame | gpl.GeoDataFrame
GeoSeries = gpd.GeoSeries | gpl.GeoSeries
DataFrame = pd.DataFrame | pl.DataFrame | GeoDataFame
Series = pd.Series | pl.Series | GeoSeries
Index = pd.Index | pl.Series
BoolSeries = pd.Series | pl.Series
MaskLike = AgnosticMask | PandasMaskLike | PolarsMaskLike
IdsLike = AgnosticIds | PandasIdsLike | PolarsIdsLike


###----- Time ------###
TimeT = float | int


###----- Space -----###
Coordinates = tuple[int, int] | tuple[float, float]
Node_ID = int
AgnosticPositionsLike = (
    Sequence[Coordinates] | Sequence[Node_ID] | Coordinates | Node_ID
)
PolarsPositionsLike = (
    AgnosticPositionsLike
    | pl.DataFrame
    | tuple[pl.Series, pl.Series]
    | gpl.GeoSeries
    | pl.Series
)
PandasPositionsLike = (
    AgnosticPositionsLike
    | pd.DataFrame
    | tuple[pd.Series, pd.Series]
    | gpd.GeoSeries
    | pd.Series
)
PositionsLike = AgnosticPositionsLike | PolarsPositionsLike | PandasPositionsLike
