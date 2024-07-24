from collections.abc import Collection, Iterator, Sequence
from typing import Literal

import numpy as np
import pandas as pd
from typing_extensions import Any

from mesa_frames.abstract.mixin import DataFrameMixin
from mesa_frames.types_ import PandasMaskLike


class PandasMixin(DataFrameMixin):
    def _df_with_columns(
        self, original_df: pd.DataFrame, new_columns: list[str], data: Any
    ) -> pd.DataFrame:
        original_df[new_columns] = data
        return original_df

    def _df_column_names(self, df: pd.DataFrame) -> list[str]:
        return df.columns.tolist() + df.index.names

    def _df_combine_first(
        self, original_df: pd.DataFrame, new_df: pd.DataFrame, index_cols: list[str]
    ) -> pd.DataFrame:
        return original_df.combine_first(new_df)

    def _df_concat(
        self,
        dfs: Collection[pd.DataFrame],
        how: Literal["horizontal"] | Literal["vertical"] = "vertical",
        ignore_index: bool = False,
    ) -> pd.DataFrame:
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
        df = pd.DataFrame(data=data, columns=columns).astype(dtypes)
        if index_col:
            df.set_index(index_col)
        return df

    def _df_contains(
        self,
        df: pd.DataFrame,
        column: str,
        values: Any | Sequence[Any],
    ) -> pd.Series:
        return pd.Series(values, index=values).isin(df[column])

    def _df_get_bool_mask(
        self,
        df: pd.DataFrame,
        index_col: str,
        mask: PandasMaskLike = None,
        negate: bool = False,
    ) -> pd.Series:
        if isinstance(mask, pd.Series) and mask.dtype == bool and len(mask) == len(df):
            result = mask
        elif isinstance(mask, pd.DataFrame):
            if mask.index.name == index_col:
                result = pd.Series(df.index.isin(mask.index), index=df.index)
            elif index_col in mask.columns:
                result = pd.Series(df.index.isin(mask[index_col]), index=df.index)
            else:
                raise ValueError(
                    f"A DataFrame mask must have a column/index with name {index_col}"
                )
        elif mask is None or mask == "all":
            result = pd.Series(True, index=df.index)
        elif isinstance(mask, Sequence):
            result = pd.Series(df.index.isin(mask), index=df.index)
        else:
            result = pd.Series(df.index.isin([mask]), index=df.index)

        if negate:
            result = ~result

        return result

    def _df_get_masked_df(
        self,
        df: pd.DataFrame,
        index_col: str,
        mask: PandasMaskLike | None = None,
        columns: list[str] | None = None,
        negate: bool = False,
    ) -> pd.DataFrame:
        b_mask = self._df_get_bool_mask(df, index_col, mask, negate)
        if columns:
            return df.loc[b_mask, columns]
        return df.loc[b_mask]

    def _df_iterator(self, df: pd.DataFrame) -> Iterator[dict[str, Any]]:
        for index, row in df.iterrows():
            row_dict = row.to_dict()
            row_dict["unique_id"] = index
            yield row_dict

    def _df_norm(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._df_constructor(
            data=[np.linalg.norm(df, axis=1), df.index],
            columns=[df.columns, df.index.name],
            index_col=df.index.name,
        )

    def _df_remove(
        self,
        df: pd.DataFrame,
        ids: Sequence[Any],
        index_col: str | None = None,
    ) -> pd.DataFrame:
        return df[~df.index.isin(ids)]

    def _df_sample(
        self,
        df: pd.DataFrame,
        n: int | None = None,
        frac: float | None = None,
        with_replacement: bool = False,
        shuffle: bool = False,
        seed: int | None = None,
    ) -> pd.DataFrame:
        return df.sample(
            n=n, frac=frac, replace=with_replacement, shuffle=shuffle, random_state=seed
        )

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
        return pd.Series(values, index=values).isin(srs)

    def _srs_range(
        self,
        name: str,
        start: int,
        end: int,
        step: int = 1,
    ) -> pd.Series:
        return pd.Series(np.arange(start, end, step), name=name)
