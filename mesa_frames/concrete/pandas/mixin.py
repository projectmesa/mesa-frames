from collections.abc import Collection, Iterator, Sequence
from typing import Literal

import numpy as np
import pandas as pd
from typing_extensions import Any

from mesa_frames.abstract.mixin import DataFrameMixin
from mesa_frames.types_ import PandasMask


class PandasMixin(DataFrameMixin):
    def _df_column_names(self, df: pd.DataFrame) -> list[str]:
        return df.columns.tolist() + df.index.names

    def _df_combine_first(
        self,
        original_df: pd.DataFrame,
        new_df: pd.DataFrame,
        index_col: str | list[str],
    ) -> pd.DataFrame:
        if index_col != original_df.index.name:
            original_df = original_df.set_index(index_col)
        if index_col != new_df.index.name:
            new_df = new_df.set_index(index_col)
        return original_df.combine_first(new_df)

    def _df_concat(
        self,
        dfs: Collection[pd.DataFrame] | Collection[pd.Series],
        how: Literal["horizontal"] | Literal["vertical"] = "vertical",
        ignore_index: bool = False,
    ) -> pd.Series | pd.DataFrame:
        return pd.concat(
            dfs, axis=0 if how == "vertical" else 1, ignore_index=ignore_index
        )

    def _df_constructor(
        self,
        data: Sequence[Sequence] | dict[str | Any] | None = None,
        columns: list[str] | None = None,
        index_col: str | list[str] | None = None,
        dtypes: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        df = pd.DataFrame(data=data, columns=columns)
        if dtypes:
            df = df.astype(dtypes)
        if index_col:
            df = df.set_index(index_col)
        return df

    def _df_contains(
        self,
        df: pd.DataFrame,
        column: str,
        values: Sequence[Any],
    ) -> pd.Series:
        return pd.Series(values, index=values).isin(df[column])

    def _df_filter(
        self,
        df: pd.DataFrame,
        condition: pd.DataFrame,
        all: bool = True,
    ) -> pd.DataFrame:
        if all and isinstance(condition, pd.DataFrame):
            return df[condition.all(axis=1)]
        return df[condition]

    def _df_get_bool_mask(
        self,
        df: pd.DataFrame,
        index_col: str,
        mask: PandasMask = None,
        negate: bool = False,
    ) -> pd.Series:
        # Get the index column
        if df.index.name == index_col:
            srs = df.index
        else:
            srs = df[index_col]
        if isinstance(mask, pd.Series) and mask.dtype == bool and len(mask) == len(df):
            mask.index = df.index
            result = mask
        elif mask is None:
            result = pd.Series(True, index=df.index)
        else:
            if isinstance(mask, pd.DataFrame):
                if mask.index.name == index_col:
                    mask = mask.index
                else:
                    mask = mask[index_col]
            elif isinstance(mask, Sequence):
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
        index_col: str,
        mask: PandasMask | None = None,
        columns: str | list[str] | None = None,
        negate: bool = False,
    ) -> pd.DataFrame:
        b_mask = self._df_get_bool_mask(df, index_col, mask, negate)
        if columns:
            return df.loc[b_mask, columns]
        return df.loc[b_mask]

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
        return left.merge(
            right,
            how=how,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
            suffixes=("", suffix),
        )

    def _df_norm(self, df: pd.DataFrame) -> pd.Series:
        return self._srs_constructor(
            np.linalg.norm(df, axis=1), name="norm", index=df.index
        )

    def _df_rename_columns(
        self,
        df: pd.DataFrame,
        old_columns: list[str],
        new_columns: list[str],
    ) -> pd.DataFrame:
        return df.rename(columns=dict(zip(old_columns, new_columns)))

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
        if isinstance(data, dict):
            return original_df.assign(**data)
        elif isinstance(data, pd.DataFrame):
            data = data.set_index(original_df.index)
            original_df.update(data)
            return original_df
        elif isinstance(data, pd.Series):
            data.index = original_df.index
        original_df[new_columns] = data
        return original_df

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
