"""Type aliases for the mesa_frames package."""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING, Annotated, Any, Literal, Union

import numpy as np
import polars as pl
from beartype.vale import IsEqual
from numpy import ndarray

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
PolarsIndex = pl.Series | pl.DataFrame
PolarsBoolSrs = pl.Series | pl.Expr
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
# Had to define this because beartype has issues with polars._typing.IntoExpr
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
Index = PolarsIndex
BoolSeries = PolarsBoolSrs
Mask = PolarsMask
AgentMask = AgentPolarsMask
IdsLike = AgnosticIds | PolarsIdsLike
ArrayLike = ndarray | Series | Sequence
Infinity = Annotated[float, IsEqual[math.inf]]  # Only accepts math.inf

from typing_extensions import TypeAliasType

# Common option types
KeyBy = Literal["name", "index", "type"]

# Selectors for choosing AgentSets at the registry level.
# Use runtime-enforceable aliases while avoiding import cycles.
# Strategy:
# - At runtime, avoid importing agentset modules (which can create import
#   cycles). Prefer lazy aliases via typing_extensions.TypeAliasType that take
#   string targets. These allow runtime validators (for example beartype) to
#   resolve names lazily instead of importing modules eagerly.

AbstractAgentSetSelector = TypeAliasType(
    "AbstractAgentSetSelector",
    (
        "mesa_frames.abstract.agentset.AbstractAgentSet | "
        "type[mesa_frames.abstract.agentset.AbstractAgentSet] | "
        "str | Collection["
        "mesa_frames.abstract.agentset.AbstractAgentSet | "
        "type[mesa_frames.abstract.agentset.AbstractAgentSet] | str] | None"
    ),
)

AgentSetSelector = TypeAliasType(
    "AgentSetSelector",
    (
        "mesa_frames.concrete.agentset.AgentSet | "
        "type[mesa_frames.concrete.agentset.AgentSet] | "
        "str | Collection["
        "mesa_frames.concrete.agentset.AgentSet | "
        "type[mesa_frames.concrete.agentset.AgentSet] | str] | None"
    ),
)

__all__ = [
    # common
    "DataFrame",
    "Series",
    "Index",
    "BoolSeries",
    "Mask",
    "AgentMask",
    "IdsLike",
    "ArrayLike",
    "KeyBy",
    # selectors
    "AbstractAgentSetSelector",
    "AgentSetSelector",
]

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
