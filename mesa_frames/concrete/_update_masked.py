"""Internal helper for applying masked updates without joins.

This module intentionally contains only low-level utilities used by concrete
implementations. It is not part of the public API.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl


class _MaskedUpdateMixin:
    """Apply updates to a Polars DataFrame given an aligned boolean mask."""

    @staticmethod
    def _reject_callables(updates: dict[str, object]) -> None:
        bad = [k for k, v in updates.items() if callable(v)]
        if bad:
            raise TypeError(
                "Callable update values are not supported; offending keys: "
                + ", ".join(bad)
            )

    @staticmethod
    def _value_requires_tmp_column(value: object) -> bool:
        return isinstance(
            value, (pl.Series, np.ndarray, list, tuple)
        ) and not isinstance(value, (str, bytes))

    def _apply_masked_updates(
        self,
        df: pl.DataFrame,
        mask: np.ndarray | pl.Series,
        updates: dict[str, object],
    ) -> pl.DataFrame:
        n_total = int(df.height)
        if isinstance(mask, np.ndarray):
            if mask.dtype != bool:
                raise TypeError("mask ndarray must be boolean")
            if int(mask.shape[0]) != n_total:
                raise ValueError("mask length mismatch")
            mask_s = pl.Series("_mf_mask", mask, dtype=pl.Boolean)
            mask_np = mask.astype(bool, copy=False)
        else:
            if mask.dtype != pl.Boolean:
                raise TypeError("mask Series must be boolean")
            if int(mask.len()) != n_total:
                raise ValueError("mask length mismatch")
            mask_s = mask
            mask_np = mask_s.to_numpy().astype(bool, copy=False)

        selected_idx = np.flatnonzero(mask_np)
        n_selected = int(selected_idx.shape[0])

        tmp_exprs: list[pl.Series] = []
        tmp_name_for_col: dict[str, str] = {}

        def _as_row_aligned_series(col: str, value: object) -> pl.Series:
            if isinstance(value, pl.Series):
                v_len = int(value.len())
                if v_len == n_total:
                    return pl.Series(col, value)
                if v_len == n_selected:
                    arr = value.to_numpy()
                    if arr.dtype.kind in {"b", "i", "u", "f"}:
                        full = np.zeros(n_total, dtype=arr.dtype)
                    else:
                        full = np.empty(n_total, dtype=arr.dtype)
                        if n_total:
                            full[:] = "" if arr.dtype.kind in {"U", "S"} else None
                    if n_selected:
                        full[selected_idx] = arr
                    return pl.Series(col, full)
                raise ValueError("Series length mismatch")
            if isinstance(value, np.ndarray):
                if value.ndim != 1:
                    raise ValueError("ndarray value must be 1D")
                v_len = int(value.shape[0])
                if v_len == n_total:
                    return pl.Series(col, value)
                if v_len == n_selected:
                    if value.dtype.kind in {"b", "i", "u", "f"}:
                        full = np.zeros(n_total, dtype=value.dtype)
                    else:
                        full = np.empty(n_total, dtype=value.dtype)
                        if n_total:
                            full[:] = "" if value.dtype.kind in {"U", "S"} else None
                    if n_selected:
                        full[selected_idx] = value
                    return pl.Series(col, full)
                raise ValueError("ndarray value must match df height or selected rows")
            if isinstance(value, (list, tuple)):
                v_len = len(value)
                if v_len == n_total:
                    return pl.Series(col, list(value))
                if v_len == n_selected:
                    arr = np.asarray(value)
                    if arr.dtype.kind in {"b", "i", "u", "f"}:
                        full = np.zeros(n_total, dtype=arr.dtype)
                    else:
                        full = np.empty(n_total, dtype=arr.dtype)
                        if n_total:
                            full[:] = "" if arr.dtype.kind in {"U", "S"} else None
                    if n_selected:
                        full[selected_idx] = arr
                    return pl.Series(col, full)
                raise ValueError("sequence length mismatch")
            raise TypeError("unsupported row-aligned value")

        # Materialize any row-aligned RHS values as temporary columns.
        for col, value in updates.items():
            if isinstance(value, pl.Expr):
                continue
            if isinstance(value, str):
                continue
            arr_like = self._value_requires_tmp_column(value)
            if arr_like:
                tmp_name = f"_mf_rhs_{col}"
                if tmp_name in df.columns:
                    raise ValueError(f"temporary column name collision: {tmp_name}")
                tmp_name_for_col[col] = tmp_name
                tmp_exprs.append(_as_row_aligned_series(tmp_name, value))

        if tmp_exprs:
            df = df.with_columns(tmp_exprs)

        exprs: list[pl.Expr] = []
        for col, value in updates.items():
            if isinstance(value, pl.Expr):
                rhs = value
            elif isinstance(value, str):
                rhs = pl.col(value) if value in df.columns else pl.lit(value)
            elif col in tmp_name_for_col:
                rhs = pl.col(tmp_name_for_col[col])
            else:
                rhs = pl.lit(value)

            if col not in df.columns:
                # Creating new columns is allowed; masked rows keep null.
                base = pl.lit(None)
            else:
                base = pl.col(col)
            exprs.append(pl.when(mask_s).then(rhs).otherwise(base).alias(col))

        out = df.with_columns(exprs)
        if tmp_name_for_col:
            out = out.drop(list(tmp_name_for_col.values()))
        return out
