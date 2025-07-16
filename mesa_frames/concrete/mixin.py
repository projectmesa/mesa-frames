"""
Polars-specific mixin for DataFrame operations in mesa-frames.

This module provides a concrete implementation of the DataFrameMixin using Polars
as the backend for DataFrame operations. It defines the PolarsMixin class, which
implements DataFrame operations specific to Polars, offering efficient data
manipulation and analysis capabilities for mesa-frames components.

Classes:
    PolarsMixin(DataFrameMixin):
        A Polars-based implementation of DataFrame operations. This class provides
        methods for manipulating and analyzing data stored in Polars LazyFrames,
        tailored for use in mesa-frames components like AgentSetPolars and GridPolars.

The PolarsMixin class is designed to be used as a mixin with other mesa-frames
classes, providing them with Polars-specific LazyFrame functionality. It implements
the abstract methods defined in the DataFrameMixin, ensuring consistent DataFrame
operations across the mesa-frames package.

Usage:
    The PolarsMixin is typically used in combination with other base classes:

    from mesa_frames.abstract import AgentSetDF
    from mesa_frames.concrete.mixin import PolarsMixin

    class AgentSetPolars(AgentSetDF, PolarsMixin):
        def __init__(self, model):
            super().__init__(model)
            self.agents = pl.LazyFrame()  # Initialize empty LazyFrame

        def some_method(self):
            # Use Polars operations provided by the mixin
            result = self._df_groupby(self.agents, 'some_column')
            # ... further processing ...

Features:
    - High-performance LazyFrame operations using Polars
    - Support for lazy evaluation with improved query optimization
    - Efficient memory usage and fast computation
    - Integration with Polars' query optimization capabilities

For more detailed information on the PolarsMixin class and its methods, refer to
the class docstring.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Hashable, Iterator, Sequence
from typing import Any, Literal, overload

import polars as pl

from mesa_frames.abstract.mixin import DataFrameMixin
from mesa_frames.types_ import PolarsDataFrameInput, PolarsIndex, PolarsMask


class PolarsMixin(DataFrameMixin):
    """Polars-specific implementation of DataFrame operations using LazyFrames."""

    # TODO: complete with other dtypes
    _dtypes_mapping: dict[str, Any] = {
        "int64": pl.Int64,
        "bool": pl.Boolean,
        "uint64": pl.UInt64,
    }

    def _df_add(
        self,
        df: pl.DataFrame,
        other: pl.DataFrame | Collection[float | int],
        axis: Literal["index"] | Literal["columns"] = "index",
        index_cols: str | list[str] | None = None,
    ) -> pl.LazyFrame:
        return self._df_operation(
            df=df,
            other=other,
            operation=lambda x, y: x + y,
            axis=axis,
            index_cols=index_cols,
        )

    def _df_all(
        self,
        df: pl.LazyFrame,
        name: str = "all",
        axis: Literal["index", "columns"] = "columns",
    ) -> pl.Expr:
        if axis == "index":
            # Return an expression that will evaluate to all values across index
            return pl.all(pl.col("*")).alias(name)
        # Return an expression for all values across columns
        return pl.all_horizontal(pl.col("*")).alias(name)

    def _df_and(
        self,
        df: pl.DataFrame,
        other: pl.DataFrame | Collection[float | int],
        axis: Literal["index"] | Literal["columns"] = "index",
        index_cols: str | list[str] | None = None,
    ) -> pl.LazyFrame:
        return self._df_operation(
            df=df,
            other=other,
            operation=lambda x, y: x & y,
            axis=axis,
            index_cols=index_cols,
        )

    def _df_column_names(self, df: pl.LazyFrame) -> list[str]:
        # This operation requires schema inspection which is available on LazyFrame
        return df.columns

    def _df_combine_first(
        self,
        original_df: pl.LazyFrame,
        new_df: pl.LazyFrame,
        index_cols: str | list[str],
    ) -> pl.LazyFrame:
        # Create a sequential index using with_row_count instead of int_range
        original_df = original_df.with_row_count("_index")
        common_cols = set(original_df.columns) & set(new_df.columns)
        merged_df = original_df.join(new_df, on=index_cols, how="full", suffix="_right")

        # Use expressions to coalesce values
        coalesce_exprs = [
            pl.coalesce(pl.col(col), pl.col(f"{col}_right")).alias(col)
            for col in common_cols
            if col in merged_df.columns and f"{col}_right" in merged_df.columns
        ]

        # Apply coalesce expressions and drop right columns
        merged_df = merged_df.with_columns(coalesce_exprs)
        right_cols = [col for col in merged_df.columns if col.endswith("_right")]
        merged_df = merged_df.drop(right_cols)

        # Sort by index and drop index column
        return merged_df.sort("_index").drop("_index")

    @overload
    def _df_concat(
        self,
        objs: Collection[pl.LazyFrame],
        how: Literal["horizontal"] | Literal["vertical"] = "vertical",
        ignore_index: bool = False,
        index_cols: str | list[str] | None = None,
    ) -> pl.LazyFrame: ...

    @overload
    def _df_concat(
        self,
        objs: Collection[pl.Expr],
        how: Literal["vertical"] = "vertical",
        ignore_index: bool = False,
        index_cols: str | list[str] | None = None,
    ) -> pl.Expr: ...

    @overload
    def _df_concat(
        self,
        objs: Collection[pl.Expr],
        how: Literal["horizontal"] = "horizontal",
        ignore_index: bool = False,
        index_cols: str | list[str] | None = None,
    ) -> pl.LazyFrame: ...

    def _df_concat(
        self,
        objs: Collection[pl.LazyFrame] | Collection[pl.Expr],
        how: Literal["horizontal"] | Literal["vertical"] = "vertical",
        ignore_index: bool = False,
        index_cols: str | None = None,
    ) -> pl.LazyFrame | pl.Expr:
        if isinstance(next(iter(objs), None), pl.LazyFrame) and how == "vertical":
            how = "diagonal_relaxed"

        if isinstance(next(iter(objs), None), pl.Expr) and how == "horizontal":
            # Convert expressions to LazyFrames for horizontal concat
            obj = pl.LazyFrame().with_columns(list(objs))
        else:
            # Use concat on LazyFrames directly
            obj = pl.concat(objs, how=how)

        if isinstance(obj, pl.LazyFrame) and how == "horizontal" and ignore_index:
            # Rename columns if ignore_index is True
            rename_dict = {c: str(i) for i, c in enumerate(obj.columns)}
            obj = obj.rename(rename_dict)

        return obj

    def _df_constructor(
        self,
        data: PolarsDataFrameInput | None = None,
        columns: list[str] | None = None,
        index: PolarsIndex | Collection[Hashable] | None = None,
        index_cols: str | list[str] | None = None,
        dtypes: dict[str, str | type] | None = None,
    ) -> pl.DataFrame:
        if dtypes is not None:
            dtypes = {k: self._dtypes_mapping.get(v, v) for k, v in dtypes.items()}

        # Create LazyFrame directly
        df = pl.LazyFrame(data=data, schema=columns, schema_overrides=dtypes)

        if index is not None:
            if index_cols is not None:
                if isinstance(index_cols, str):
                    index_cols = [index_cols]
                index_df = pl.LazyFrame({col: index for col in index_cols})
            else:
                index_df = pl.LazyFrame({"index": index})

            if len(df.schema) == 0:
                # Empty LazyFrame case
                df = index_df
            else:
                # Use cross join for single row df or regular join otherwise
                df = index_df.join(df, how="cross")

        return df

    def _df_contains(
        self,
        df: pl.LazyFrame,
        column: str,
        values: Collection[Any],
    ) -> pl.Series:
        return pl.Series("contains", values).is_in(df[column])

    def _df_div(
        self,
        df: pl.DataFrame,
        other: pl.DataFrame | Collection[float | int],
        axis: Literal["index"] | Literal["columns"] = "index",
        index_cols: str | list[str] | None = None,
    ) -> pl.LazyFrame:
        return self._df_operation(
            df=df,
            other=other,
            operation=lambda x, y: x / y,
            axis=axis,
            index_cols=index_cols,
        )

    def _df_drop_columns(
        self,
        df: pl.LazyFrame,
        columns: str | list[str],
    ) -> pl.LazyFrame:
        return df.drop(columns)

    def _df_drop_duplicates(
        self,
        df: pl.LazyFrame,
        subset: str | list[str] | None = None,
        keep: Literal["first", "last", False] = "first",
    ) -> pl.LazyFrame:
        # If subset is None, use all columns
        if subset is None:
            subset = df.columns

        if keep == "first":
            return df.unique(subset=subset, keep="first")
        elif keep == "last":
            return df.unique(subset=subset, keep="last")
        else:
            # For keep=False, drop all duplicates
            return df.filter(~pl.col(subset).is_duplicated())

    def _df_ge(
        self,
        df: pl.DataFrame,
        other: pl.DataFrame | Collection[float | int],
        axis: Literal["index", "columns"] = "index",
        index_cols: str | list[str] | None = None,
    ) -> pl.LazyFrame:
        return self._df_operation(
            df=df,
            other=other,
            operation=lambda x, y: x >= y,
            axis=axis,
            index_cols=index_cols,
        )

    def _df_get_bool_mask(
        self,
        df: pl.LazyFrame,
        index_cols: str | list[str] | None = None,
        mask: PolarsMask = None,
        negate: bool = False,
    ) -> pl.Expr:
        def bool_mask_from_expr(mask: pl.Expr) -> pl.Expr:
            return mask

        def bool_mask_from_lazyframe(mask: pl.LazyFrame) -> pl.Expr:
            if index_cols is None:
                raise ValueError(
                    "index_cols must be provided when using LazyFrame mask"
                )

            if isinstance(index_cols, str):
                return pl.col(index_cols).is_in(mask.select(index_cols))
            else:
                # For multiple index columns, create an expression to check if in the mask
                join_cols = [pl.col(col) for col in index_cols]
                return pl.struct(join_cols).is_in(mask.select(index_cols))

        def bool_mask_from_values(values) -> pl.Expr:
            if index_cols is None:
                raise ValueError("index_cols must be provided when using value mask")

            if isinstance(index_cols, str):
                return pl.col(index_cols).is_in(values)
            else:
                # This is simplified and may need adjustment for multi-column case
                raise NotImplementedError(
                    "Multi-column masking with raw values not implemented"
                )

        if isinstance(mask, pl.Expr):
            result = bool_mask_from_expr(mask)
        elif isinstance(mask, pl.LazyFrame):
            result = bool_mask_from_lazyframe(mask)
        elif mask is None or mask == "all":
            result = pl.lit(True)
        elif isinstance(mask, Collection):
            result = bool_mask_from_values(mask)
        else:
            result = bool_mask_from_values([mask])

        if negate:
            result = ~result

        return result

    def _df_get_masked_df(
        self,
        df: pl.LazyFrame,
        index_cols: str | list[str] | None = None,
        mask: PolarsMask | None = None,
        columns: list[str] | None = None,
        negate: bool = False,
    ) -> pl.LazyFrame:
        b_mask = self._df_get_bool_mask(df, index_cols, mask, negate=negate)
        if columns:
            return df.filter(b_mask).select(columns)
        return df.filter(b_mask)

    def _df_groupby_cumcount(
        self, df: pl.LazyFrame, by: str | list[str], name="cum_count"
    ) -> pl.Expr:
        return pl.cumcount().over(by).alias(name)

    @overload
    def _df_index(self, df: pl.DataFrame, index_col: str) -> pl.Series: ...

    @overload
    def _df_index(
        self, df: pl.DataFrame, index_col: Collection[str]
    ) -> pl.DataFrame: ...

    def _df_index(
        self, df: pl.DataFrame, index_col: str | Collection[str]
    ) -> pl.Series | pl.DataFrame:
        return df[index_col]

    def _df_iterator(self, df: pl.LazyFrame) -> Iterator[dict[str, Any]]:
        return iter(df.collect().iter_rows(named=True))

    def _df_join(
        self,
        left: pl.LazyFrame,
        right: pl.LazyFrame,
        index_cols: str | list[str] | None = None,
        on: str | list[str] | None = None,
        left_on: str | list[str] | None = None,
        right_on: str | list[str] | None = None,
        how: (
            Literal["left"]
            | Literal["right"]
            | Literal["inner"]
            | Literal["outer"]
            | Literal["cross"]
        ) = "left",
        suffix="_right",
    ) -> pl.LazyFrame:
        if how == "outer":
            how = "full"
        if how == "right":
            left, right = right, left
            left_on, right_on = right_on, left_on
            how = "left"
        return left.join(
            right, on=on, left_on=left_on, right_on=right_on, how=how, suffix=suffix
        )

    def _df_lt(
        self,
        df: pl.DataFrame,
        other: pl.DataFrame | Collection[float | int],
        axis: Literal["index", "columns"] = "index",
        index_cols: str | list[str] | None = None,
    ) -> pl.LazyFrame:
        return self._df_operation(
            df=df,
            other=other,
            operation=lambda x, y: x < y,
            axis=axis,
            index_cols=index_cols,
        )

    def _df_mod(
        self,
        df: pl.DataFrame,
        other: pl.DataFrame | Collection[float | int],
        axis: Literal["index"] | Literal["columns"] = "index",
        index_cols: str | list[str] | None = None,
    ) -> pl.LazyFrame:
        return self._df_operation(
            df=df,
            other=other,
            operation=lambda x, y: x % y,
            axis=axis,
            index_cols=index_cols,
        )

    def _df_mul(
        self,
        df: pl.DataFrame,
        other: pl.DataFrame | Collection[float | int],
        axis: Literal["index", "columns"] = "index",
        index_cols: str | list[str] | None = None,
    ) -> pl.LazyFrame:
        return self._df_operation(
            df=df,
            other=other,
            operation=lambda x, y: x * y,
            axis=axis,
            index_cols=index_cols,
        )

    @overload
    def _df_norm(
        self,
        df: pl.LazyFrame,
        srs_name: str = "norm",
        include_cols: Literal[False] = False,
    ) -> pl.Expr: ...

    @overload
    def _df_norm(
        self,
        df: pl.Expr,
        srs_name: str = "norm",
        include_cols: Literal[True] = True,
    ) -> pl.LazyFrame: ...

    def _df_norm(
        self,
        df: pl.LazyFrame,
        srs_name: str = "norm",
        include_cols: bool = False,
    ) -> pl.Expr | pl.LazyFrame:
        srs = (
            df.with_columns(pl.col("*").pow(2)).sum_horizontal().sqrt().alias(srs_name)
        )
        if include_cols:
            return df.with_columns(srs)
        return srs

    def _df_operation(
        self,
        df: pl.DataFrame,
        other: pl.DataFrame | Collection[float | int],
        operation: Callable[[pl.Expr, pl.Expr], pl.Expr],
        axis: Literal["index", "columns"] = "index",
        index_cols: str | list[str] | None = None,
    ) -> pl.LazyFrame:
        if isinstance(other, pl.LazyFrame):
            if index_cols is not None:
                # Join with the other LazyFrame
                op_df = df.join(other, how="left", on=index_cols, suffix="_op")
            else:
                # Without index cols, assume matching order and do a horizontal concat
                other = other.rename({col: f"{col}_op" for col in other.columns})
                op_df = pl.concat([df, other], how="horizontal")

            # Apply the operation to matching columns
            expr_list = []
            for col in df.columns:
                if col not in (index_cols or []):
                    if f"{col}_op" in op_df.columns:
                        expr_list.append(
                            operation(pl.col(col), pl.col(f"{col}_op")).alias(col)
                        )
                    else:
                        expr_list.append(pl.col(col))
                else:
                    expr_list.append(pl.col(col))

            return op_df.with_columns(expr_list).select(df.columns)
        elif isinstance(other, (Sequence, pl.Series)):
            if axis == "index":
                # Apply operation row-wise
                if isinstance(other, pl.Series):
                    # Convert Series to an expression
                    other_expr = pl.lit(other.to_list())
                else:
                    other_expr = pl.lit(list(other))

                expr_list = [
                    operation(pl.col(col), other_expr).alias(col) for col in df.columns
                ]
                return df.with_columns(expr_list)
            else:
                # Apply operation column-wise
                expr_list = [
                    operation(pl.col(col), pl.lit(other[i])).alias(col)
                    for i, col in enumerate(df.columns)
                ]
                return df.with_columns(expr_list)
        else:
            raise ValueError("other must be a LazyFrame or a Sequence")

    def _df_or(
        self,
        df: pl.DataFrame,
        other: pl.DataFrame | Collection[float | int],
        axis: Literal["index"] | Literal["columns"] = "index",
        index_cols: str | list[str] | None = None,
    ) -> pl.LazyFrame:
        return self._df_operation(
            df=df,
            other=other,
            operation=lambda x, y: x | y,
            axis=axis,
            index_cols=index_cols,
        )

    def _df_reindex(
        self,
        df: pl.DataFrame,
        other: Sequence[Hashable] | pl.DataFrame | pl.Series,
        new_index_cols: str | list[str],
        original_index_cols: str | list[str] | None = None,
    ) -> pl.LazyFrame:
        # If other is a LazyFrame, extract the index columns
        if isinstance(other, pl.LazyFrame):
            other = other.select(new_index_cols)
        else:
            # If other is a sequence, create a LazyFrame with it
            other = pl.LazyFrame({new_index_cols: other})

        # Perform a left join to reindex
        if original_index_cols is None:
            original_index_cols = new_index_cols
        result = other.join(
            df, left_on=new_index_cols, right_on=original_index_cols, how="left"
        )
        return result

    def _df_rename_columns(
        self, df: pl.LazyFrame, old_columns: list[str], new_columns: list[str]
    ) -> pl.LazyFrame:
        return df.rename(dict(zip(old_columns, new_columns)))

    def _df_reset_index(
        self,
        df: pl.LazyFrame,
        index_cols: str | list[str] | None = None,
        drop: bool = False,
    ) -> pl.LazyFrame:
        if drop and index_cols is not None:
            return df.drop(index_cols)
        else:
            return df

    def _df_sample(
        self,
        df: pl.LazyFrame,
        n: int | None = None,
        frac: float | None = None,
        with_replacement: bool = False,
        shuffle: bool = False,
        seed: int | None = None,
    ) -> pl.LazyFrame:
        return df.sample(
            n=n,
            fraction=frac,
            with_replacement=with_replacement,
            shuffle=shuffle,
            seed=seed,
        )

    def _df_set_index(
        self,
        df: pl.DataFrame,
        index_name: str | Collection[str],
        new_index: Collection[Hashable] | None = None,
    ) -> pl.DataFrame:
        if new_index is None:
            return df
        return df.with_columns(**{index_name: new_index})

    def _df_with_columns(
        self,
        original_df: pl.LazyFrame,
        data: Sequence | pl.LazyFrame | Sequence[Sequence] | dict[str | Any] | Any,
        new_columns: str | list[str] | None = None,
    ) -> pl.LazyFrame:
        if (
            (isinstance(data, Sequence) and isinstance(data[0], Sequence))
            or isinstance(
                data, pl.LazyFrame
            )  # Currently, pl.LazyFrame is not a Sequence
            or (
                isinstance(data, dict)
                and isinstance(data[list(data.keys())[0]], Sequence)
            )
        ):
            # This means that data is a Sequence of Sequences (rows)
            data = pl.LazyFrame(data, new_columns)
            original_df = original_df.select(pl.exclude(data.columns))
            return original_df.hstack(data)
        if not isinstance(data, dict):
            assert new_columns is not None, "new_columns must be specified"
            if isinstance(new_columns, list):
                data = {col: value for col, value in zip(new_columns, data)}
            else:
                data = {new_columns: data}
            return original_df.with_columns(**data)

    def _srs_constructor(
        self,
        data: Collection[Any] | None = None,
        name: str | None = None,
        dtype: str | None = None,
        index: Sequence[Any] | None = None,
    ) -> pl.Series:
        if dtype is not None:
            dtype = self._dtypes_mapping[dtype]
        return pl.Series(name=name, values=list(data), dtype=dtype)

    def _srs_contains(
        self,
        srs: Collection[Any],
        values: Any | Sequence[Any],
    ) -> pl.Expr:
        if not isinstance(values, Collection):
            values = [values]
        return pl.Series(values).is_in(pl.Series(srs))

    def _srs_range(
        self,
        name: str,
        start: int,
        end: int,
        step: int = 1,
    ) -> pl.Series:
        return pl.arange(start=start, end=end, step=step, eager=True).alias(name)

    def _srs_to_df(
        self, srs: pl.Series, index: pl.Series | None = None
    ) -> pl.LazyFrame:
        df = srs.to_frame().lazy()
        if index:
            return df.with_columns({index.name: index})
        return df
