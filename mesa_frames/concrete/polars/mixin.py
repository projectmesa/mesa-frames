from collections.abc import Collection, Iterator, Sequence
from typing import Literal

import polars as pl
from typing_extensions import Any, overload

from collections.abc import Hashable

from mesa_frames.abstract.mixin import DataFrameMixin
from mesa_frames.types_ import PolarsMask


class PolarsMixin(DataFrameMixin):
    # TODO: complete with other dtypes
    _dtypes_mapping: dict[str, Any] = {"int64": pl.Int64, "bool": pl.Boolean}

    def _df_add(
        self,
        df: pl.DataFrame,
        other: pl.DataFrame | Sequence[float | int],
        axis: Literal["index"] | Literal["columns"] = "index",
        index_cols: str | list[str] | None = None,
    ) -> pl.DataFrame:
        if isinstance(other, pl.DataFrame):
            if axis == "index":
                if index_cols is None:
                    raise ValueError(
                        "index_cols must be specified when axis is 'index'"
                    )
                return (
                    df.join(other.select(pl.all().suffix("_add")), on=index_cols)
                    .with_columns(
                        [
                            (pl.col(col) + pl.col(f"{col}_add")).alias(col)
                            for col in df.columns
                            if col not in index_cols
                        ]
                    )
                    .select(df.columns)
                )
            else:
                return df.select(
                    [
                        (pl.col(col) + pl.col(other.columns[i])).alias(col)
                        for i, col in enumerate(df.columns)
                    ]
                )
        elif isinstance(other, Sequence):
            if axis == "index":
                other_series = pl.Series("addend", other)
                return df.with_columns(
                    [(pl.col(col) + other_series).alias(col) for col in df.columns]
                )
            else:
                return df.with_columns(
                    [
                        (pl.col(col) + other[i]).alias(col)
                        for i, col in enumerate(df.columns)
                    ]
                )
        else:
            raise ValueError("other must be a DataFrame or a Sequence")

    def _df_all(
        self,
        df: pl.DataFrame,
        name: str = "all",
        axis: Literal["index", "columns"] = "columns",
    ) -> pl.Series:
        if axis == "columns":
            return df.select(pl.col("*").all()).to_series()
        return df.with_columns(all=pl.all_horizontal())["all"]

    def _df_column_names(self, df: pl.DataFrame) -> list[str]:
        return df.columns

    def _df_combine_first(
        self,
        original_df: pl.DataFrame,
        new_df: pl.DataFrame,
        index_cols: str | list[str],
    ) -> pl.DataFrame:
        new_df = original_df.join(new_df, on=index_cols, how="full", suffix="_right")
        # Find columns with the _right suffix and update the corresponding original columns
        updated_columns = []
        for col in new_df.columns:
            if col.endswith("_right"):
                original_col = col.replace("_right", "")
                updated_columns.append(
                    pl.when(pl.col(col).is_not_null())
                    .then(pl.col(col))
                    .otherwise(pl.col(original_col))
                    .alias(original_col)
                )

        # Apply the updates and remove the _right columns
        new_df = new_df.with_columns(updated_columns).select(
            pl.col(r"^(?!.*_right$).*")
        )
        return new_df

    @overload
    def _df_concat(
        self,
        objs: Collection[pl.DataFrame],
        how: Literal["horizontal"] | Literal["vertical"] = "vertical",
        ignore_index: bool = False,
        index_cols: str | None = None,
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
        return pl.concat(
            objs, how="vertical_relaxed" if how == "vertical" else "horizontal_relaxed"
        )

    def _df_constructor(
        self,
        data: Sequence[Sequence] | dict[str | Any] | None = None,
        columns: list[str] | None = None,
        index: Sequence[Hashable] | None = None,
        index_cols: str | list[str] | None = None,
        dtypes: dict[str, str] | None = None,
    ) -> pl.DataFrame:
        dtypes = {k: self._dtypes_mapping.get(v, v) for k, v in dtypes.items()}
        return pl.DataFrame(data=data, schema=dtypes if dtypes else columns)

    def _df_contains(
        self,
        df: pl.DataFrame,
        column: str,
        values: Sequence[Any],
    ) -> pl.Series:
        return pl.Series(values, index=values).is_in(df[column])

    def _df_div(
        self,
        df: pl.DataFrame,
        other: pl.DataFrame | pl.Series | Sequence[float | int],
        axis: Literal["index"] | Literal["columns"] = "index",
        index_cols: str | list[str] | None = None,
    ) -> pl.DataFrame:
        if isinstance(other, pl.DataFrame):
            if axis == "index":
                if index_cols is None:
                    raise ValueError(
                        "index_cols must be specified when axis is 'index'"
                    )
                return (
                    df.join(other.select(pl.all().suffix("_div")), on=index_cols)
                    .with_columns(
                        [
                            (pl.col(col) / pl.col(f"{col}_div")).alias(col)
                            for col in df.columns
                            if col not in index_cols
                        ]
                    )
                    .select(df.columns)
                )
            else:  # axis == "columns"
                return df.select(
                    [
                        (pl.col(col) / pl.col(other.columns[i])).alias(col)
                        for i, col in enumerate(df.columns)
                    ]
                )
        elif isinstance(other, pl.Series):
            if axis == "index":
                return df.with_columns(
                    [
                        (pl.col(col) / other).alias(col)
                        for col in df.columns
                        if col != other.name
                    ]
                )
            else:  # axis == "columns"
                return df.with_columns(
                    [
                        (pl.col(col) / other[i]).alias(col)
                        for i, col in enumerate(df.columns)
                    ]
                )
        elif isinstance(other, Sequence):
            if axis == "index":
                other_series = pl.Series("divisor", other)
                return df.with_columns(
                    [(pl.col(col) / other_series).alias(col) for col in df.columns]
                )
            else:  # axis == "columns"
                return df.with_columns(
                    [
                        (pl.col(col) / other[i]).alias(col)
                        for i, col in enumerate(df.columns)
                    ]
                )
        else:
            raise ValueError("other must be a DataFrame, Series, or Sequence")

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
        # If subset is a string, convert it to a list
        elif isinstance(subset, str):
            subset = [subset]

        # Determine the sort order based on 'keep'
        if keep == "first":
            sort_expr = [pl.col(col).rank("dense", reverse=True) for col in subset]
        elif keep == "last":
            sort_expr = [pl.col(col).rank("dense") for col in subset]
        elif keep is False:
            # If keep is False, we don't need to sort, just group and filter
            return df.group_by(subset).agg(pl.all().first()).sort(subset)
        else:
            raise ValueError("'keep' must be either 'first', 'last', or False")

        # Add a rank column, sort by it, and keep only the first row of each group
        return (
            df.with_columns(pl.struct(sort_expr).alias("__rank"))
            .sort("__rank")
            .group_by(subset)
            .agg(pl.all().first())
            .sort(subset)
            .drop("__rank")
        )

    def _df_filter(
        self,
        df: pl.DataFrame,
        condition: pl.Series,
        all: bool = True,
    ) -> pl.DataFrame:
        if all:
            return df.filter(pl.all(condition))
        return df.filter(condition)

    def _df_get_bool_mask(
        self,
        df: pl.DataFrame,
        index_cols: str | list[str],
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
            return df[index_cols].is_in(mask)

        if isinstance(mask, pl.Expr):
            result = mask
        elif isinstance(mask, pl.Series):
            result = bool_mask_from_series(mask)
        elif isinstance(mask, pl.DataFrame):
            if index_cols in mask.columns:
                result = bool_mask_from_series(mask[index_cols])
            elif len(mask.columns) == 1 and mask.dtypes[0] == pl.Boolean:
                result = bool_mask_from_series(mask[mask.columns[0]])
            else:
                raise KeyError(
                    f"DataFrame must have an {index_cols} column or a single boolean column."
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
        index_cols: str,
        mask: PolarsMask | None = None,
        columns: list[str] | None = None,
        negate: bool = False,
    ) -> pl.DataFrame:
        b_mask = self._df_get_bool_mask(df, index_cols, mask, negate=negate)
        if columns:
            return df.filter(b_mask)[columns]
        return df.filter(b_mask)

    def _df_groupby_cumcount(self, df: pl.DataFrame, by: str | list[str]) -> pl.Series:
        return df.with_columns(pl.col(by).cum_count().alias("cumcount"))

    def _df_iterator(self, df: pl.DataFrame) -> Iterator[dict[str, Any]]:
        return iter(df.iter_rows(named=True))

    def _df_join(
        self,
        left: pl.DataFrame,
        right: pl.DataFrame,
        on: str | list[str] | None = None,
        left_on: str | list[str] | None = None,
        right_on: str | list[str] | None = None,
        how: Literal["left"]
        | Literal["right"]
        | Literal["inner"]
        | Literal["outer"]
        | Literal["cross"] = "left",
        suffix="_right",
    ) -> pl.DataFrame:
        return left.join(
            right,
            on=on,
            left_on=left_on,
            right_on=right_on,
            how=how,
            lsuffix="",
            rsuffix=suffix,
        )

    def _df_mul(
        self,
        df: pl.DataFrame,
        other: pl.DataFrame | Sequence[float | int],
        axis: Literal["index", "columns"] = "index",
        index_cols: str | list[str] | None = None,
    ) -> pl.DataFrame:
        if isinstance(other, pl.DataFrame):
            if axis == "index":
                if index_cols is None:
                    raise ValueError(
                        "index_cols must be specified when axis is 'index'"
                    )
                return (
                    df.join(other.select(pl.all().suffix("_mul")), on=index_cols)
                    .with_columns(
                        [
                            (pl.col(col) * pl.col(f"{col}_mul")).alias(col)
                            for col in df.columns
                            if col not in index_cols
                        ]
                    )
                    .select(df.columns)
                )
            else:  # axis == "columns"
                return df.select(
                    [
                        (pl.col(col) * pl.col(other.columns[i])).alias(col)
                        for i, col in enumerate(df.columns)
                    ]
                )
        elif isinstance(other, Sequence):
            if axis == "index":
                other_series = pl.Series("multiplier", other)
                return df.with_columns(
                    [(pl.col(col) * other_series).alias(col) for col in df.columns]
                )
            else:
                return df.with_columns(
                    [
                        (pl.col(col) * other[i]).alias(col)
                        for i, col in enumerate(df.columns)
                    ]
                )
        else:
            raise ValueError("other must be a DataFrame or a Sequence")

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
            df.with_columns(pl.col("*").pow(2).alias("*"))
            .sum_horizontal()
            .sqrt()
            .rename(srs_name)
        )
        if include_cols:
            return df.with_columns(srs_name=srs)
        return srs

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
        if drop:
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
            n=n, frac=frac, replace=with_replacement, shuffle=shuffle, seed=seed
        )

    def _df_set_index(
        self,
        df: pl.DataFrame,
        index_name: str,
        new_index: Sequence[Hashable] | None = None,
    ) -> pl.DataFrame:
        if new_index is None:
            return df

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
        dtype: Any | None = None,
        index: Sequence[Any] | None = None,
    ) -> pl.Series:
        return pl.Series(name=name, values=data, dtype=self._dtypes_mapping[dtype])

    def _srs_contains(
        self,
        srs: Sequence[Any],
        values: Any | Sequence[Any],
    ) -> pl.Series:
        return pl.Series(values, index=values).is_in(srs)

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
