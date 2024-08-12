from collections.abc import Callable, Collection, Hashable, Iterator, Sequence
from typing import Literal

import numpy as np
import pandas as pd
from typing_extensions import Any, overload

from mesa_frames.abstract.mixin import DataFrameMixin
from mesa_frames.types_ import PandasMask


class PandasMixin(DataFrameMixin):
    def _df_add(
        self,
        df: pd.DataFrame,
        other: pd.DataFrame | Sequence[float | int],
        axis: Literal["index", "columns"] = "index",
        index_cols: str | list[str] | None = None,
    ) -> pd.DataFrame:
        return df.add(other=other, axis=axis)

    def _df_and(
        self,
        df: pd.DataFrame,
        other: pd.DataFrame | Sequence[float | int],
        axis: Literal["index"] | Literal["columns"] = "index",
        index_cols: str | list[str] | None = None,
    ) -> pd.DataFrame:
        return self._df_logical_operation(
            df=df,
            other=other,
            operation=lambda x, y: x & y,
            axis=axis,
            index_cols=index_cols,
        )

    def _df_all(
        self,
        df: pd.DataFrame,
        name: str = "all",
        axis: str = "columns",
    ) -> pd.Series:
        return df.all(axis).rename(name)

    def _df_column_names(self, df: pd.DataFrame) -> list[str]:
        return df.columns.tolist() + df.index.names

    def _df_combine_first(
        self,
        original_df: pd.DataFrame,
        new_df: pd.DataFrame,
        index_cols: str | list[str],
    ) -> pd.DataFrame:
        if (isinstance(index_cols, str) and index_cols != original_df.index.name) or (
            isinstance(index_cols, list) and index_cols != original_df.index.names
        ):
            original_df = original_df.set_index(index_cols)

        if (isinstance(index_cols, str) and index_cols != original_df.index.name) or (
            isinstance(index_cols, list) and index_cols != original_df.index.names
        ):
            new_df = new_df.set_index(index_cols)
        return original_df.combine_first(new_df)

    @overload
    def _df_concat(
        self,
        objs: Collection[pd.DataFrame],
        how: Literal["horizontal"] | Literal["vertical"] = "vertical",
        ignore_index: bool = False,
        index_cols: str | None = None,
    ) -> pd.DataFrame: ...

    @overload
    def _df_concat(
        self,
        objs: Collection[pd.Series],
        how: Literal["horizontal"] = "horizontal",
        ignore_index: bool = False,
        index_cols: str | None = None,
    ) -> pd.DataFrame: ...

    @overload
    def _df_concat(
        self,
        objs: Collection[pd.Series],
        how: Literal["vertical"] = "vertical",
        ignore_index: bool = False,
        index_cols: str | None = None,
    ) -> pd.Series: ...

    def _df_concat(
        self,
        objs: Collection[pd.DataFrame] | Collection[pd.Series],
        how: Literal["horizontal"] | Literal["vertical"] = "vertical",
        ignore_index: bool = False,
        index_cols: str | None = None,
    ) -> pd.Series | pd.DataFrame:
        df = pd.concat(
            objs, axis=0 if how == "vertical" else 1, ignore_index=ignore_index
        )
        if index_cols:
            return df.set_index(index_cols)
        return df

    def _df_constructor(
        self,
        data: Sequence[Sequence] | dict[str | Any] | None = None,
        columns: list[str] | None = None,
        index: Sequence[Hashable] | None = None,
        index_cols: str | list[str] | None = None,
        dtypes: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        df = pd.DataFrame(data=data, columns=columns, index=index)
        if dtypes:
            df = df.astype(dtypes)
        if index_cols:
            df = df.set_index(index_cols)
        return df

    def _df_contains(
        self,
        df: pd.DataFrame,
        column: str,
        values: Sequence[Any],
    ) -> pd.Series:
        if df.index.name == column:
            return pd.Series(values).isin(df.index)
        return pd.Series(values).isin(df[column])

    def _df_div(
        self,
        df: pd.DataFrame,
        other: pd.DataFrame | Sequence[float | int],
        axis: Literal["index", "columns"] = "index",
        index_cols: str | list[str] | None = None,
    ) -> pd.DataFrame:
        return df.div(other=other, axis=axis)

    def _df_drop_columns(
        self,
        df: pd.DataFrame,
        columns: str | list[str],
    ) -> pd.DataFrame:
        return df.drop(columns=columns)

    def _df_drop_duplicates(
        self,
        df: pd.DataFrame,
        subset: str | list[str] | None = None,
        keep: Literal["first", "last", False] = "first",
    ) -> pd.DataFrame:
        return df.drop_duplicates(subset=subset, keep=keep)

    def _df_ge(
        self,
        df: pd.DataFrame,
        other: pd.DataFrame | Sequence[float | int],
        axis: Literal["index", "columns"] = "index",
        index_cols: str | list[str] | None = None,
    ) -> pd.DataFrame:
        return df.ge(other, axis=axis)

    def _df_get_bool_mask(
        self,
        df: pd.DataFrame,
        index_cols: str | list[str] | None = None,
        mask: PandasMask = None,
        negate: bool = False,
    ) -> pd.Series:
        # Get the index column
        if (isinstance(index_cols, str) and df.index.name == index_cols) or (
            isinstance(index_cols, list) and df.index.names == index_cols
        ):
            srs = df.index
        elif index_cols is not None:
            srs = df.set_index(index_cols).index
        if isinstance(mask, pd.Series) and mask.dtype == bool and len(mask) == len(df):
            mask.index = df.index
            result = mask
        elif mask is None:
            result = pd.Series(True, index=df.index)
        else:
            if isinstance(mask, pd.DataFrame):
                if (isinstance(index_cols, str) and mask.index.name == index_cols) or (
                    isinstance(index_cols, list) and mask.index.names == index_cols
                ):
                    mask = mask.index
                else:
                    mask = mask.set_index(index_cols).index

            elif isinstance(mask, Collection):
                pass
            else:  # single value
                mask = [mask]
            result = pd.Series(srs.isin(mask), index=df.index)
        if negate:
            result = ~result
        return result

    def _df_get_masked_df(
        self,
        df: pd.DataFrame,
        index_cols: str | list[str] | None = None,
        mask: PandasMask | None = None,
        columns: str | list[str] | None = None,
        negate: bool = False,
    ) -> pd.DataFrame:
        b_mask = self._df_get_bool_mask(df, index_cols, mask, negate)
        if columns:
            return df.loc[b_mask, columns]
        return df.loc[b_mask]

    def _df_groupby_cumcount(
        self, df: pd.DataFrame, by: str | list[str], name: str = "cum_count"
    ) -> pd.Series:
        return df.groupby(by).cumcount().rename(name)

    def _df_iterator(self, df: pd.DataFrame) -> Iterator[dict[str, Any]]:
        for index, row in df.iterrows():
            row_dict = row.to_dict()
            if df.index.name:
                row_dict[df.index.name] = index
            else:
                row_dict["index"] = index
            yield row_dict

    def _df_join(
        self,
        left: pd.DataFrame,
        right: pd.DataFrame,
        index_cols: str | list[str] | None = None,
        on: str | list[str] | None = None,
        left_on: str | list[str] | None = None,
        right_on: str | list[str] | None = None,
        how: Literal["left"]
        | Literal["right"]
        | Literal["inner"]
        | Literal["outer"]
        | Literal["cross"] = "left",
        suffix="_right",
    ) -> pd.DataFrame:
        left_index = False
        right_index = False
        if on:
            left_on = on
            right_on = on
        if left.index.name and left.index.name == left_on:
            left_index = True
            left_on = None
        if right.index.name and right.index.name == right_on:
            right_index = True
            right_on = None
        # Reset index if it is not used as a key to keep it in the DataFrame
        if not left_index and left.index.name:
            left = left.reset_index()
        if not right_index and right.index.name:
            right = right.reset_index()
        df = left.merge(
            right,
            how=how,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
            suffixes=("", suffix),
        )
        if index_cols:
            return df.set_index(index_cols)
        else:
            return df

    def _df_lt(
        self,
        df: pd.DataFrame,
        other: pd.DataFrame | Sequence[float | int],
        axis: Literal["index", "columns"] = "index",
        index_cols: str | list[str] | None = None,
    ) -> pd.DataFrame:
        return df.lt(other, axis=axis)

    def _df_logical_operation(
        self,
        df: pd.DataFrame,
        other: pd.DataFrame | Sequence[bool],
        operation: Callable[
            [pd.DataFrame, Sequence[bool] | pd.DataFrame], pd.DataFrame
        ],
        axis: Literal["index"] | Literal["columns"] = "index",
        index_cols: str | list[str] | None = None,
    ) -> pd.DataFrame:
        if isinstance(other, pd.DataFrame):
            if index_cols is not None:
                if df.index.name != index_cols:
                    df = df.set_index(index_cols)
                if other.index.name != index_cols:
                    other = other.set_index(index_cols)
            other = other.reindex(df.index, fill_value=np.nan)
            return operation(df, other)
        else:  # Sequence[bool]
            other = pd.Series(other)
            if axis == "index":
                other.index = df.index
                return operation(df, other.values[:, None]).astype(bool)
            else:
                return operation(df, other.values[None, :]).astype(bool)

    def _df_mul(
        self,
        df: pd.DataFrame,
        other: pd.DataFrame | Sequence[float | int],
        axis: Literal["index", "columns"] = "index",
        index_cols: str | list[str] | None = None,
    ) -> pd.DataFrame:
        return df.mul(other=other, axis=axis)

    @overload
    def _df_norm(
        self,
        df: pd.DataFrame,
        srs_name: str = "norm",
        include_cols: Literal[False] = False,
    ) -> pd.Series: ...

    @overload
    def _df_norm(
        self,
        df: pd.DataFrame,
        srs_name: str = "norm",
        include_cols: Literal[True] = True,
    ) -> pd.DataFrame: ...

    def _df_norm(
        self,
        df: pd.DataFrame,
        srs_name: str = "norm",
        include_cols: bool = False,
    ) -> pd.Series | pd.DataFrame:
        srs = self._srs_constructor(
            np.linalg.norm(df, axis=1), name=srs_name, index=df.index
        )
        if include_cols:
            return self._df_with_columns(df, srs, srs_name)
        else:
            return srs

    def _df_or(
        self,
        df: pd.DataFrame,
        other: pd.DataFrame | Sequence[bool],
        axis: Literal["index"] | Literal["columns"] = "index",
        index_cols: str | list[str] | None = None,
    ) -> pd.DataFrame:
        return self._df_logical_operation(
            df=df,
            other=other,
            operation=lambda x, y: x | y,
            axis=axis,
            index_cols=index_cols,
        )

    def _df_rename_columns(
        self,
        df: pd.DataFrame,
        old_columns: list[str],
        new_columns: list[str],
    ) -> pd.DataFrame:
        return df.rename(columns=dict(zip(old_columns, new_columns)))

    def _df_reset_index(
        self,
        df: pd.DataFrame,
        index_cols: str | list[str] | None = None,
        drop: bool = False,
    ) -> pd.DataFrame:
        return df.reset_index(level=index_cols, drop=drop)

    def _df_sample(
        self,
        df: pd.DataFrame,
        n: int | None = None,
        frac: float | None = None,
        with_replacement: bool = False,
        shuffle: bool = False,
        seed: int | None = None,
    ) -> pd.DataFrame:
        return df.sample(n=n, frac=frac, replace=with_replacement, random_state=seed)

    def _df_set_index(
        self,
        df: pd.DataFrame,
        index_name: str,
        new_index: Sequence[Hashable] | None = None,
    ) -> pd.DataFrame:
        if new_index is None:
            df = df.set_index(index_name)
        else:
            df = df.set_index(new_index)
            df.index.name = index_name
        return df

    def _df_with_columns(
        self,
        original_df: pd.DataFrame,
        data: pd.DataFrame
        | pd.Series
        | Sequence[Sequence]
        | dict[str | Any]
        | Sequence[Any]
        | Any,
        new_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        df = original_df.copy()
        if isinstance(data, dict):
            return df.assign(**data)
        elif isinstance(data, pd.DataFrame):
            data = data.set_index(df.index)
            new_columns = data.columns
        elif isinstance(data, pd.Series):
            data.index = df.index
        df.loc[:, new_columns] = data
        return df

    def _srs_constructor(
        self,
        data: Sequence[Sequence] | None = None,
        name: str | None = None,
        dtype: Any | None = None,
        index: Sequence[Any] | None = None,
    ) -> pd.Series:
        return pd.Series(data, name=name, dtype=dtype, index=index)

    def _srs_contains(
        self, srs: Sequence[Any], values: Any | Sequence[Any]
    ) -> pd.Series:
        if isinstance(values, Sequence):
            return pd.Series(values, index=values).isin(srs)
        else:
            return pd.Series(values, index=[values]).isin(srs)

    def _srs_range(
        self,
        name: str,
        start: int,
        end: int,
        step: int = 1,
    ) -> pd.Series:
        return pd.Series(np.arange(start, end, step), name=name)

    def _srs_to_df(self, srs: pd.Series, index: pd.Index | None = None) -> pd.DataFrame:
        df = srs.to_frame()
        if index:
            return df.set_index(index)
        return df
