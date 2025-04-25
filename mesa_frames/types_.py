"""Type aliases for the mesa_frames package."""

from __future__ import annotations
from collections.abc import Collection, Sequence
from datetime import date, datetime, time, timedelta
from typing import Literal, Annotated, Union, Any
from collections.abc import Mapping
from beartype.vale import IsEqual
import math
import polars as pl
from numpy import ndarray
import numpy as np
# import geopolars as gpl # TODO: Uncomment when geopolars is available

###----- Optional Types -----###
try:
    import pandas as pd

    PandasDataFrame = pd.DataFrame
except ImportError:
    # just give us a class so annotations don’t break
    PandasDataFrame = type("PandasDataFrame", (), {})
    PandasSeries = type("PandasSeries", (), {})

try:
    import pyarrow as pa

    ArrowTable = pa.Table
except ImportError:
    ArrowTable = type("ArrowTable", (), {})


####----- Agnostic Types -----####
AgnosticMask = (
    Any | Sequence[Any] | None
)  # Any is a placeholder for any type if it's a single value
AgnosticAgentMask = Sequence[int] | int | Literal["all", "active"] | None
AgnosticIds = int | Collection[int]

###----- Polars Types -----###
PolarsDataFrameInput = Union[
    Mapping[
        str, Union[Sequence[object], Mapping[str, Sequence[object]], pl.Series, Any]
    ],
    Sequence[Any],
    np.ndarray,
    ArrowTable,
    pd.DataFrame,
]
PolarsMask = pl.Expr | pl.Series | pl.DataFrame | AgnosticMask
AgentPolarsMask = AgnosticAgentMask | pl.Expr | pl.Series | pl.DataFrame | Sequence[int]
PolarsIdsLike = AgnosticIds | pl.Series | pl.DataFrame
PolarsGridCapacity = list[pl.Expr]
IntoExpr = (
    int
    | float
    | date
    | time
    | datetime
    | timedelta
    | str
    | bool
    | bytes
    | list[Any]
    | pl.Expr
    | pl.Series
    | None
)

###----- Generic -----###
# GeoDataFrame = gpd.GeoDataFrame | gpl.GeoDataFrame
DataFrame = pl.DataFrame
DataFrameInput = PolarsDataFrameInput
Series = pl.Series
Index = pl.Series
BoolSeries = pl.Series | pl.Expr
Mask = PolarsMask
AgentMask = AgentPolarsMask
IdsLike = AgnosticIds | PolarsIdsLike
ArrayLike = ndarray | Series | Sequence
Infinity = Annotated[float, IsEqual[math.inf]]  # Only accepts math.inf

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

DiscreteSpaceCapacity = GridCapacity | NetworkCapacity | np.ndarray
