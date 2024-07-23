from collections.abc import Collection, Iterator, Sequence
from typing import Literal

import polars as pl
from typing_extensions import Any

from mesa_frames.abstract.mixin import DataFrameMixin
from mesa_frames.types_ import PolarsMaskLike


class PolarsMixin(DataFrameMixin):
    # TODO: complete with other dtypes
    _dtypes_mapping: dict[str, Any] = {"int64": pl.Int64, "bool": pl.Boolean}

    def _df_add_columns(
        self, original_df: pl.DataFrame, new_columns: list[str], data: Any
    ) -> pl.DataFrame:
        return original_df.with_columns(
            **{col: value for col, value in zip(new_columns, data)}
        )

    def _df_combine_first(
        self, original_df: pl.DataFrame, new_df: pl.DataFrame, index_cols: list[str]
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

    def _df_concat(
        self,
        dfs: Collection[pl.DataFrame],
        how: Literal["horizontal"] | Literal["vertical"] = "vertical",
        ignore_index: bool = False,
    ) -> pl.DataFrame:
        return pl.concat(
            dfs, how="vertical_relaxed" if how == "vertical" else "horizontal_relaxed"
        )

    def _df_constructor(
        self,
        data: Sequence[Sequence] | dict[str | Any] | None = None,
        columns: list[str] | None = None,
        index_col: str | list[str] | None = None,
        dtypes: dict[str, str] | None = None,
    ) -> pl.DataFrame:
        dtypes = {k: self._dtypes_mapping.get(v, v) for k, v in dtypes.items()}
        return pl.DataFrame(data=data, schema=dtypes if dtypes else columns)

    def _df_get_bool_mask(
        self,
        df: pl.DataFrame,
        index_col: str,
        mask: PolarsMaskLike = None,
        negate: bool = False,
    ) -> pl.Series | pl.Expr:
        def bool_mask_from_series(mask: pl.Series) -> pl.Series:
            if (
                isinstance(mask, pl.Series)
                and mask.dtype == pl.Boolean
                and len(mask) == len(df)
            ):
                return mask
            return df[index_col].is_in(mask)

        if isinstance(mask, pl.Expr):
            result = mask
        elif isinstance(mask, pl.Series):
            result = bool_mask_from_series(mask)
        elif isinstance(mask, pl.DataFrame):
            if index_col in mask.columns:
                result = bool_mask_from_series(mask[index_col])
            elif len(mask.columns) == 1 and mask.dtypes[0] == pl.Boolean:
                result = bool_mask_from_series(mask[mask.columns[0]])
            else:
                raise KeyError(
                    f"DataFrame must have an {index_col} column or a single boolean column."
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
        index_col: str,
        mask: PolarsMaskLike | None = None,
        columns: list[str] | None = None,
        negate: bool = False,
    ) -> pl.DataFrame:
        b_mask = self._df_get_bool_mask(df, index_col, mask, negate=negate)
        if columns:
            return df.filter(b_mask)[columns]
        return df.filter(b_mask)

    def _df_iterator(self, df: pl.DataFrame) -> Iterator[dict[str, Any]]:
        return iter(df.iter_rows(named=True))

    def _df_norm(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.col("*").pow(2).alias("*")).sum_horizontal().sqrt()

    def _df_remove(
        self, df: pl.DataFrame, ids: Sequence[Any], index_col: str | None = None
    ) -> pl.DataFrame:
        return df.filter(pl.col(index_col).is_in(ids).not_())

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

    def _srs_constructor(
        self,
        data: Sequence[Any] | None = None,
        name: str | None = None,
        dtype: Any | None = None,
        index: Sequence[Any] | None = None,
    ) -> pl.Series:
        return pl.Series(name=name, values=data, dtype=self._dtypes_mapping[dtype])
