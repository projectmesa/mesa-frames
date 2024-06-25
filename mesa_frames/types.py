from typing import Collection, Hashable, Literal

####----- Agnostic Types -----####
AgnosticMask = Literal["all", "active"] | Hashable | None
AgnosticIds = int | Collection[int]

###----- Pandas Types -----###
import pandas as pd
from numpy import ndarray

ArrayLike = pd.api.extensions.ExtensionArray | ndarray
AnyArrayLike = ArrayLike | pd.Index | pd.Series
PandasMaskLike = AgnosticMask | pd.Series | pd.DataFrame | AnyArrayLike
PandasIdsLike = AgnosticIds | pd.Series[int] | pd.Index

###----- Polars Types -----###
import polars as pl

PolarsMaskLike = AgnosticMask | pl.Expr | pl.Series | pl.DataFrame | Collection[int]
PolarsIdsLike = AgnosticIds | pl.Series

###----- Generic -----###

DataFrame = pd.DataFrame | pl.DataFrame
Series = pd.Series | pl.Series
BoolSeries = pd.Series | pl.Series
MaskLike = AgnosticMask | PandasMaskLike | PolarsMaskLike
IdsLike = AgnosticIds | pd.Series[int] | pd.Index | pl.Series
