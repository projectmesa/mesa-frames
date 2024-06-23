from typing import Collection, Hashable, Literal

####----- Agnostic Types -----####
AgnosticMask = Literal["all", "active"] | Hashable | None


###----- Pandas Types -----###
import pandas as pd
from numpy import ndarray

ArrayLike = pd.api.extensions.ExtensionArray | ndarray
AnyArrayLike = ArrayLike | pd.Index | pd.Series
PandasMaskLike = AgnosticMask | pd.Series | pd.DataFrame | AnyArrayLike


###----- Polars Types -----###
import polars as pl

PolarsMaskLike = AgnosticMask | pl.Expr | pl.Series | pl.DataFrame | Collection[int]


###----- Generic -----###

DataFrame = pd.DataFrame | pl.DataFrame
Series = pd.Series | pl.Series
BoolSeries = pd.Series | pl.Series | pl.Expr
MaskLike = AgnosticMask | PandasMaskLike | PolarsMaskLike
