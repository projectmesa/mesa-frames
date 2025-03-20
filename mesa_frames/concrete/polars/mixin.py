"""
Polars-specific mixin for DataFrame operations in mesa-frames.

This module provides a concrete implementation of the DataFrameMixin using Polars
as the backend for DataFrame operations. It defines the PolarsMixin class, which
implements DataFrame operations specific to Polars, offering efficient data
manipulation and analysis capabilities for mesa-frames components.

Classes:
    PolarsMixin(DataFrameMixin):
        A Polars-based implementation of DataFrame operations. This class provides
        methods for manipulating and analyzing data stored in Polars DataFrames,
        tailored for use in mesa-frames components like AgentSetPolars and GridPolars.

The PolarsMixin class is designed to be used as a mixin with other mesa-frames
classes, providing them with Polars-specific DataFrame functionality. It implements
the abstract methods defined in the DataFrameMixin, ensuring consistent DataFrame
operations across the mesa-frames package.

Usage:
    The PolarsMixin is typically used in combination with other base classes:

    from mesa_frames.abstract import AgentSetDF
    from mesa_frames.concrete.polars.mixin import PolarsMixin

    class AgentSetPolars(AgentSetDF, PolarsMixin):
        def __init__(self, model):
            super().__init__(model)
            self.agents = pl.DataFrame()  # Initialize empty DataFrame

        def some_method(self):
            # Use Polars operations provided by the mixin
            result = self._df_groupby(self.agents, 'some_column')
            # ... further processing ...

Features:
    - High-performance DataFrame operations using Polars
    - Support for both eager and lazy evaluation
    - Efficient memory usage and fast computation
    - Integration with Polars' query optimization capabilities

For more detailed information on the PolarsMixin class and its methods, refer to
the class docstring.
"""

from collections.abc import Callable, Collection, Hashable, Iterator, Sequence
from typing import Literal

import polars as pl
from typing_extensions import Any, overload

from mesa_frames.abstract.mixin import DataFrameMixin
from mesa_frames.types_ import DataFrame, PolarsMask


class PolarsMixin(DataFrameMixin):
    """Polars-specific implementation of DataFrame operations."""

    # TODO: complete with other dtypes
    _dtypes_mapping: dict[str, Any] = {"int64": pl.Int64, "bool": pl.Boolean}

    def _df_add(
        self,
        df: pl.DataFrame,
        other: pl.DataFrame | Sequence[float | int],
        axis: Literal["index"] | Literal["columns"] = "index",
        index_cols: str | list[str] | None = None,
    ) -> pl.DataFrame:
        return self._df_operation(
            df=df,
            other=other,
            operation=lambda x, y: x + y,
            axis=axis,
            index_cols=index_cols,
        )

    def _df_all(
        self,
        df: pl.DataFrame,
        name: str = "all",
        axis: Literal["index", "columns"] = "columns",
    ) -> pl.Series:
        if axis == "index":
            return pl.Series(name, df.select(pl.col("*").all()).row(0))
        return df.with_columns(pl.all_horizontal("*").alias(name))[name]

    def _df_and(
        self,
        df: pl.DataFrame,
        other: pl.DataFrame | Sequence[float | int],
        axis: Literal["index"] | Literal["columns"] = "index",
        index_cols: str | list[str] | None = None,
    ) -> pl.DataFrame:
        return self._df_operation(
            df=df,
            other=other,
            operation=lambda x, y: x & y,
            axis=axis,
            index_cols=index_cols,
        )

    def _df_column_names(self, df: pl.DataFrame) -> list[str]:
        return df.columns

    def _df_combine_first(
        self,
        original_df: pl.DataFrame,
        new_df: pl.DataFrame,
        index_cols: str | list[str],
    ) -> pl.DataFrame:
        original_df = original_df.with_columns(_index=pl.int_range(0, len(original_df)))
        common_cols = set(original_df.columns) & set(new_df.columns)
        merged_df = original_df.join(new_df, on=index_cols, how="full", suffix="_right")
        merged_df = (
            merged_df.with_columns(
                pl.coalesce(pl.col(col), pl.col(f"{col}_right")).alias(col)
                for col in common_cols
            )
            .select(pl.exclude("^.*_right$"))
            .sort("_index")
            .drop("_index")
        )
        return merged_df

    @overload
    def _df_concat(
        self,
        objs: Collection[pl.DataFrame],
        how: Literal["horizontal"] | Literal["vertical"] = "vertical",
        ignore_index: bool = False,
        index_cols: str | list[str] | None = None,
    ) -> pl.DataFrame: ...

    @overload
    def _df_concat(
        self,
        objs: Collection[pl.Series],
        how: Literal["vertical"] = "vertical",
        ignore_index: bool = False,
        index_cols: str | list[str] | None = None,
    ) -> pl.Series: ...

    @overload
    def _df_concat(
        self,
        objs: Collection[pl.Series],
        how: Literal["horizontal"] = "horizontal",
        ignore_index: bool = False,
        index_cols: str | list[str] | None = None,
    ) -> pl.DataFrame: ...

    def _df_concat(
        self,
        objs: Collection[pl.DataFrame] | Collection[pl.Series],
        how: Literal["horizontal"] | Literal["vertical"] = "vertical",
        ignore_index: bool = False,
        index_cols: str | None = None,
    ) -> pl.Series | pl.DataFrame:
        if isinstance(objs[0], pl.DataFrame) and how == "vertical":
            how = "diagonal_relaxed"
        if isinstance(objs[0], pl.Series) and how == "horizontal":
            obj = pl.DataFrame().hstack(objs, in_place=True)
        else:
            obj = pl.concat(objs, how=how)
        if isinstance(obj, pl.DataFrame) and how == "horizontal" and ignore_index:
            obj = obj.rename(
                {c: str(i) for c, i in zip(obj.columns, range(len(obj.columns)))}
            )
        return obj

    def _df_constructor(
        self,
        data: dict[str | Any] | Sequence[Sequence] | DataFrame | None = None,
        columns: list[str] | None = None,
        index: Sequence[Hashable] | None = None,
        index_cols: str | list[str] | None = None,
        dtypes: dict[str, str] | None = None,
    ) -> pl.DataFrame:
        if dtypes is not None:
            dtypes = {k: self._dtypes_mapping.get(v, v) for k, v in dtypes.items()}

        df = pl.DataFrame(
            data=data, schema=columns, schema_overrides=dtypes, orient="row"
        )
        if index is not None:
            if index_cols is not None:
                if isinstance(index_cols, str):
                    index_cols = [index_cols]
                index_df = pl.DataFrame(index, index_cols)
            else:
                index_df = pl.DataFrame(index)
            if len(df) != len(index_df) and len(df) == 1:
                df = index_df.join(df, how="cross")
            else:
                df = index_df.hstack(df)
        return df

    def _df_contains(
        self,
        df: pl.DataFrame,
        column: str,
        values: Sequence[Any],
    ) -> pl.Series:
        return pl.Series("contains", values).is_in(df[column])

    def _df_div(
        self,
        df: pl.DataFrame,
        other: pl.DataFrame | Sequence[float | int],
        axis: Literal["index"] | Literal["columns"] = "index",
        index_cols: str | list[str] | None = None,
    ) -> pl.DataFrame:
        return self._df_operation(
            df=df,
            other=other,
            operation=lambda x, y: x / y,
            axis=axis,
            index_cols=index_cols,
        )

    def _df_drop_columns(
        self,
        df: pl.DataFrame,
        columns: str | list[str],
    ) -> pl.DataFrame:
        return df.drop(columns)

    def _df_drop_duplicates(
        self,
        df: pl.DataFrame,
        subset: str | list[str] | None = None,
        keep: Literal["first", "last", False] = "first",
    ) -> pl.DataFrame:
        # If subset is None, use all columns
        if subset is None:
            subset = df.columns
        original_col_order = df.columns
        if keep == "first":
            return (
                df.group_by(subset, maintain_order=True)
                .first()
                .select(original_col_order)
            )
        elif keep == "last":
            return (
                df.group_by(subset, maintain_order=True)
                .last()
                .select(original_col_order)
            )
        else:
            return (
                df.with_columns(pl.len().over(subset))
                .filter(pl.col("len") < 2)
                .drop("len")
                .select(original_col_order)
            )

    def _df_ge(
        self,
        df: pl.DataFrame,
        other: pl.DataFrame | Sequence[float | int],
        axis: Literal["index", "columns"] = "index",
        index_cols: str | list[str] | None = None,
    ) -> pl.DataFrame:
        return self._df_operation(
            df=df,
            other=other,
            operation=lambda x, y: x >= y,
            axis=axis,
            index_cols=index_cols,
        )

    def _df_get_bool_mask(
        self,
        df: pl.DataFrame,
        index_cols: str | list[str] | None = None,
        mask: PolarsMask = None,
        negate: bool = False,
    ) -> pl.Series | pl.Expr:
        def bool_mask_from_series(mask: pl.Series) -> pl.Series:
            if (
                isinstance(mask, pl.Series)
                and mask.dtype == pl.Boolean
                and len(mask) == len(df)
            ):
                return mask
            assert isinstance(index_cols, str)
            return df[index_cols].is_in(mask)

        def bool_mask_from_df(mask: pl.DataFrame) -> pl.Series:
            assert index_cols, list[str]
            mask = mask[index_cols].unique()
            mask = mask.with_columns(in_it=True)
            return df.join(mask, on=index_cols, how="left")["in_it"].fill_null(False)

        if isinstance(mask, pl.Expr):
            result = mask
        elif isinstance(mask, pl.Series):
            result = bool_mask_from_series(mask)
        elif isinstance(mask, pl.DataFrame):
            if index_cols in mask.columns:
                result = bool_mask_from_series(mask[index_cols])
            elif all(col in mask.columns for col in index_cols):
                result = bool_mask_from_df(mask[index_cols])
            elif len(mask.columns) == 1 and mask.dtypes[0] == pl.Boolean:
                result = bool_mask_from_series(mask[mask.columns[0]])
            else:
                raise KeyError(
                    f"Mask must have {index_cols} column(s) or a single boolean column."
                )
        elif mask is None or mask == "all":
            result = pl.Series([True] * len(df))
        elif isinstance(mask, Collection):
            result = bool_mask_from_series(pl.Series(mask))
        else:
            result = bool_mask_from_series(pl.Series([mask]))

        if negate:
            result = ~result

        return result

    def _df_get_masked_df(
        self,
        df: pl.DataFrame,
        index_cols: str | list[str] | None = None,
        mask: PolarsMask | None = None,
        columns: list[str] | None = None,
        negate: bool = False,
    ) -> pl.DataFrame:
        b_mask = self._df_get_bool_mask(df, index_cols, mask, negate=negate)
        if columns:
            return df.filter(b_mask)[columns]
        return df.filter(b_mask)

    def _df_groupby_cumcount(
        self, df: pl.DataFrame, by: str | list[str], name="cum_count"
    ) -> pl.Series:
        return df.with_columns(pl.cum_count(by).over(by).alias(name))[name]

    def _df_index(self, df: pl.DataFrame, index_col: str | list[str]) -> pl.Series:
        return df[index_col]

    def _df_iterator(self, df: pl.DataFrame) -> Iterator[dict[str, Any]]:
        return iter(df.iter_rows(named=True))

    def _df_join(
        self,
        left: pl.DataFrame,
        right: pl.DataFrame,
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
    ) -> pl.DataFrame:
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
        other: pl.DataFrame | Sequence[float | int],
        axis: Literal["index", "columns"] = "index",
        index_cols: str | list[str] | None = None,
    ) -> pl.DataFrame:
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
        other: pl.DataFrame | Sequence[float | int],
        axis: Literal["index"] | Literal["columns"] = "index",
        index_cols: str | list[str] | None = None,
    ) -> pl.DataFrame:
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
        other: pl.DataFrame | Sequence[float | int],
        axis: Literal["index", "columns"] = "index",
        index_cols: str | list[str] | None = None,
    ) -> pl.DataFrame:
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
        df: pl.DataFrame,
        srs_name: str = "norm",
        include_cols: Literal[False] = False,
    ) -> pl.Series: ...

    @overload
    def _df_norm(
        self,
        df: pl.Series,
        srs_name: str = "norm",
        include_cols: Literal[True] = True,
    ) -> pl.DataFrame: ...

    def _df_norm(
        self,
        df: pl.DataFrame,
        srs_name: str = "norm",
        include_cols: bool = False,
    ) -> pl.Series | pl.DataFrame:
        srs = (
            df.with_columns(pl.col("*").pow(2)).sum_horizontal().sqrt().rename(srs_name)
        )
        if include_cols:
            return df.with_columns(srs)
        return srs

    def _df_operation(
        self,
        df: pl.DataFrame,
        other: pl.DataFrame | Sequence[float | int],
        operation: Callable[[pl.Expr, pl.Expr], pl.Expr],
        axis: Literal["index", "columns"] = "index",
        index_cols: str | list[str] | None = None,
    ) -> pl.DataFrame:
        if isinstance(other, pl.DataFrame):
            if index_cols is not None:
                op_df = df.join(other, how="left", on=index_cols, suffix="_op")
            else:
                assert len(df) == len(
                    other
                ), "DataFrames must have the same length if index_cols is not specified"
                index_cols = []
                other = other.rename(lambda col: col + "_op")
                op_df = pl.concat([df, other], how="horizontal")
            return op_df.with_columns(
                operation(pl.col(col), pl.col(f"{col}_op")).alias(col)
                for col in df.columns
                if col not in index_cols
            ).select(df.columns)
        elif isinstance(
            other, (Sequence, pl.Series)
        ):  # Currently, pl.Series is not a Sequence
            if axis == "index":
                assert len(df) == len(
                    other
                ), "Sequence must have the same length as df if axis is 'index'"
                other_series = pl.Series("operand", other)
                return df.with_columns(
                    operation(pl.col(col), other_series).alias(col)
                    for col in df.columns
                )
            else:
                assert len(df.columns) == len(
                    other
                ), "Sequence must have the same length as df.columns if axis is 'columns'"
                return df.with_columns(
                    operation(pl.col(col), pl.lit(other[i])).alias(col)
                    for i, col in enumerate(df.columns)
                )
        else:
            raise ValueError("other must be a DataFrame or a Sequence")

    def _df_or(
        self,
        df: pl.DataFrame,
        other: pl.DataFrame | Sequence[float | int],
        axis: Literal["index"] | Literal["columns"] = "index",
        index_cols: str | list[str] | None = None,
    ) -> pl.DataFrame:
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
        other: Sequence[Hashable] | pl.DataFrame,
        new_index_cols: str | list[str],
        original_index_cols: str | list[str] | None = None,
    ) -> pl.DataFrame:
        # If other is a DataFrame, extract the index columns
        if isinstance(other, pl.DataFrame):
            other = other.select(new_index_cols)
        else:
            # If other is a sequence, create a DataFrame with it
            other = pl.Series(name=new_index_cols, values=other).to_frame()

        # Perform a left join to reindex
        if original_index_cols is None:
            original_index_cols = new_index_cols
        result = other.join(
            df, left_on=new_index_cols, right_on=original_index_cols, how="left"
        )
        return result

    def _df_rename_columns(
        self, df: pl.DataFrame, old_columns: list[str], new_columns: list[str]
    ) -> pl.DataFrame:
        return df.rename(dict(zip(old_columns, new_columns)))

    def _df_reset_index(
        self,
        df: pl.DataFrame,
        index_cols: str | list[str] | None = None,
        drop: bool = False,
    ) -> pl.DataFrame:
        if drop and index_cols is not None:
            return df.drop(index_cols)
        else:
            return df

    def _df_sample(
        self,
        df: pl.DataFrame,
        n: int | None = None,
        frac: float | None = None,
        with_replacement: bool = False,
        shuffle: bool = False,
        seed: int | None = None,
    ) -> pl.DataFrame:
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
        index_name: str,
        new_index: Sequence[Hashable] | None = None,
    ) -> pl.DataFrame:
        if new_index is None:
            return df
        return df.with_columns(**{index_name: new_index})

    def _df_with_columns(
        self,
        original_df: pl.DataFrame,
        data: Sequence | pl.DataFrame | Sequence[Sequence] | dict[str | Any] | Any,
        new_columns: str | list[str] | None = None,
    ) -> pl.DataFrame:
        if (
            (isinstance(data, Sequence) and isinstance(data[0], Sequence))
            or isinstance(
                data, pl.DataFrame
            )  # Currently, pl.DataFrame is not a Sequence
            or (
                isinstance(data, dict)
                and isinstance(data[list(data.keys())[0]], Sequence)
            )
        ):
            # This means that data is a Sequence of Sequences (rows)
            data = pl.DataFrame(data, new_columns)
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
        data: Sequence[Any] | None = None,
        name: str | None = None,
        dtype: str | None = None,
        index: Sequence[Any] | None = None,
    ) -> pl.Series:
        if dtype is not None:
            dtype = self._dtypes_mapping[dtype]
        return pl.Series(name=name, values=data, dtype=dtype)

    def _srs_contains(
        self,
        srs: Collection[Any],
        values: Any | Sequence[Any],
    ) -> pl.Series:
        if not isinstance(values, Collection):
            values = [values]
        return pl.Series(values).is_in(srs)

    def _srs_range(
        self,
        name: str,
        start: int,
        end: int,
        step: int = 1,
    ) -> pl.Series:
        return pl.arange(start=start, end=end, step=step, eager=True).rename(name)

    def _srs_to_df(
        self, srs: pl.Series, index: pl.Series | None = None
    ) -> pl.DataFrame:
        df = srs.to_frame()
        if index:
            return df.with_columns({index.name: index})
        return df
