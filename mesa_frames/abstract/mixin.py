"""
Mixin classes for mesa-frames abstract components.

This module defines mixin classes that provide common functionality and interfaces
for various components in the mesa-frames extension. These mixins are designed to
be used with the abstract base classes to create flexible and extensible
implementations.

Classes:
    CopyMixin(ABC):
        A mixin class that provides a fast copy method for classes that inherit it.
        This is useful for creating efficient copies of large data structures, such
        as DataFrames containing agent data.

    DataFrameMixin(ABC):
        A mixin class that defines an interface for DataFrame operations. This mixin
        provides a common set of methods that should be implemented by concrete
        backend classes (e.g., pandas or Polars implementations) to ensure consistent
        DataFrame manipulation across the mesa-frames package.

These mixin classes are not meant to be instantiated directly. Instead, they should
be inherited alongside other base classes to add specific functionality or to
enforce a common interface.

Usage:
    Mixin classes are typically used in multiple inheritance scenarios:

    from mesa_frames.abstract.mixin import CopyMixin, DataFrameMixin

    class MyDataFrameClass(SomeBaseClass, CopyMixin, DataFrameMixin):
        def __init__(self):
            super().__init__()
            # Implementation

        # Implement abstract methods from DataFrameMixin

Note:
    The DataFrameMixin uses Python's @abstractmethod decorator for its methods,
    ensuring that classes inheriting from it must implement these methods.

Attributes and methods of each mixin class are documented in their respective
docstrings.
"""

from abc import ABC, abstractmethod
from collections.abc import Collection, Hashable, Iterator, Sequence
from copy import copy, deepcopy
from typing import Literal

from typing_extensions import Any, Self, overload

from mesa_frames.types_ import BoolSeries, DataFrame, Index, Mask, Series
from beartype import beartype


@beartype
class CopyMixin(ABC):
    """A mixin class that provides a fast copy method for the class that inherits it."""

    _copy_with_method: dict[str, tuple[str, list[str]]] = {}
    _copy_only_reference: list[str] = [
        "_model",
    ]

    @abstractmethod
    def __init__(self): ...

    def copy(
        self,
        deep: bool = False,
        memo: dict | None = None,
        skip: list[str] | None = None,
    ) -> Self:
        """Create a copy of the Class.

        Parameters
        ----------
        deep : bool, optional
            Flag indicating whether to perform a deep copy of the AgentContainer.
            If True, all attributes of the AgentContainer will be recursively copied (except attributes in self._copy_reference_only).
            If False, only the top-level attributes will be copied.
            Defaults to False.
        memo : dict | None, optional
            A dictionary used to track already copied objects during deep copy.
            Defaults to None.
        skip : list[str] | None, optional
            A list of attribute names to skip during the copy process.
            Defaults to None.

        Returns
        -------
        Self
            A new instance of the AgentContainer class that is a copy of the original instance.
        """
        cls = self.__class__
        obj = cls.__new__(cls)

        if skip is None:
            skip = []

        if deep:
            if not memo:
                memo = {}
            memo[id(self)] = obj
            attributes = self.__dict__.copy()
            [
                setattr(obj, k, deepcopy(v, memo))
                for k, v in attributes.items()
                if k not in self._copy_with_method
                and k not in self._copy_only_reference
                and k not in skip
            ]
        else:
            [
                setattr(obj, k, copy(v))
                for k, v in self.__dict__.items()
                if k not in self._copy_with_method
                and k not in self._copy_only_reference
                and k not in skip
            ]

        # Copy attributes with a reference only
        for attr in self._copy_only_reference:
            setattr(obj, attr, getattr(self, attr))

        # Copy attributes with a specified method
        for attr in self._copy_with_method:
            attr_obj = getattr(self, attr)
            attr_copy_method, attr_copy_args = self._copy_with_method[attr]
            setattr(obj, attr, getattr(attr_obj, attr_copy_method)(*attr_copy_args))

        return obj

    def _get_obj(self, inplace: bool) -> Self:
        """Get the object to perform operations on.

        Parameters
        ----------
        inplace : bool
            If inplace, return self. Otherwise, return a copy.

        Returns
        -------
        Self
            The object to perform operations on.
        """
        if inplace:
            return self
        else:
            return deepcopy(self)

    def __copy__(self) -> Self:
        """Create a shallow copy of the AgentContainer.

        Returns
        -------
        Self
            A shallow copy of the AgentContainer.
        """
        return self.copy(deep=False)

    def __deepcopy__(self, memo: dict) -> Self:
        """Create a deep copy of the AgentContainer.

        Parameters
        ----------
        memo : dict
            A dictionary to store the copied objects.

        Returns
        -------
        Self
            A deep copy of the AgentContainer.
        """
        return self.copy(deep=True, memo=memo)


@beartype
class DataFrameMixin(ABC):
    """A mixin class which defines an interface for DataFrame operations. Most methods are abstract and should be implemented by the concrete backend."""

    def _df_remove(self, df: DataFrame, mask: Mask, index_cols: str) -> DataFrame:
        return self._df_get_masked_df(df, index_cols, mask, negate=True)

    @abstractmethod
    def _df_and(
        self,
        df: DataFrame,
        other: DataFrame | Sequence[float | int],
        axis: Literal["index", "columns"] = "index",
        index_cols: str | list[str] | None = None,
    ) -> DataFrame: ...

    @abstractmethod
    def _df_add(
        self,
        df: DataFrame,
        other: DataFrame | Sequence[float | int],
        axis: Literal["index", "columns"] = "index",
        index_cols: str | list[str] | None = None,
    ) -> DataFrame: ...

    @abstractmethod
    def _df_all(
        self,
        df: DataFrame,
        name: str = "all",
        axis: str = "columns",
    ) -> Series: ...

    @abstractmethod
    def _df_column_names(self, df: DataFrame) -> list[str]: ...

    @abstractmethod
    def _df_combine_first(
        self, original_df: DataFrame, new_df: DataFrame, index_cols: str | list[str]
    ) -> DataFrame: ...

    @overload
    @abstractmethod
    def _df_concat(
        self,
        objs: Collection[Series],
        how: Literal["vertical"] = "vertical",
        ignore_index: bool = False,
        index_cols: str | None = None,
    ) -> Series: ...

    @overload
    @abstractmethod
    def _df_concat(
        self,
        objs: Collection[Series],
        how: Literal["horizontal"] = "horizontal",
        ignore_index: bool = False,
        index_cols: str | None = None,
    ) -> DataFrame: ...

    @overload
    @abstractmethod
    def _df_concat(
        self,
        objs: Collection[DataFrame],
        how: Literal["horizontal"] | Literal["vertical"] = "vertical",
        ignore_index: bool = False,
        index_cols: str | None = None,
    ) -> DataFrame: ...

    @abstractmethod
    def _df_concat(
        self,
        objs: Collection[DataFrame] | Collection[Series],
        how: Literal["horizontal"] | Literal["vertical"] = "vertical",
        ignore_index: bool = False,
        index_cols: str | None = None,
    ) -> DataFrame | Series: ...

    @abstractmethod
    def _df_contains(
        self,
        df: DataFrame,
        column: str,
        values: Sequence[Any],
    ) -> BoolSeries: ...

    @abstractmethod
    def _df_constructor(
        self,
        data: Sequence[Sequence] | dict[str | Any] | DataFrame | None = None,
        columns: list[str] | None = None,
        index: Index | None = None,
        index_cols: str | list[str] | None = None,
        dtypes: dict[str, Any] | None = None,
    ) -> DataFrame: ...

    @abstractmethod
    def _df_div(
        self,
        df: DataFrame,
        other: DataFrame | Sequence[float | int],
        axis: Literal["index", "columns"] = "index",
        index_cols: str | list[str] | None = None,
    ) -> DataFrame: ...

    @abstractmethod
    def _df_drop_columns(
        self,
        df: DataFrame,
        columns: str | list[str],
    ) -> DataFrame: ...

    @abstractmethod
    def _df_drop_duplicates(
        self,
        df: DataFrame,
        subset: str | list[str] | None = None,
        keep: Literal["first", "last", False] = "first",
    ) -> DataFrame: ...

    @abstractmethod
    def _df_ge(
        self,
        df: DataFrame,
        other: DataFrame | Sequence[float | int],
        axis: Literal["index", "columns"] = "index",
        index_cols: str | list[str] | None = None,
    ) -> DataFrame: ...

    @abstractmethod
    def _df_get_bool_mask(
        self,
        df: DataFrame,
        index_cols: str | list[str] | None = None,
        mask: Mask | None = None,
        negate: bool = False,
    ) -> BoolSeries: ...

    @abstractmethod
    def _df_get_masked_df(
        self,
        df: DataFrame,
        index_cols: str | list[str] | None = None,
        mask: Mask | None = None,
        columns: str | list[str] | None = None,
        negate: bool = False,
    ) -> DataFrame: ...

    @abstractmethod
    def _df_groupby_cumcount(
        self, df: DataFrame, by: str | list[str], name: str = "cum_count"
    ) -> Series: ...

    @abstractmethod
    def _df_index(self, df: DataFrame, index_name: str) -> Index: ...

    @abstractmethod
    def _df_iterator(self, df: DataFrame) -> Iterator[dict[str, Any]]: ...

    @abstractmethod
    def _df_join(
        self,
        left: DataFrame,
        right: DataFrame,
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
    ) -> DataFrame: ...

    @abstractmethod
    def _df_lt(
        self,
        df: DataFrame,
        other: DataFrame | Sequence[float | int],
        axis: Literal["index", "columns"] = "index",
        index_cols: str | list[str] | None = None,
    ) -> DataFrame: ...

    @abstractmethod
    def _df_mod(
        self,
        df: DataFrame,
        other: DataFrame | Sequence[float | int],
        axis: Literal["index", "columns"] = "index",
        index_cols: str | list[str] | None = None,
    ) -> DataFrame: ...

    @abstractmethod
    def _df_mul(
        self,
        df: DataFrame,
        other: DataFrame | Sequence[float | int],
        axis: Literal["index", "columns"] = "index",
        index_cols: str | list[str] | None = None,
    ) -> DataFrame: ...

    @abstractmethod
    @overload
    def _df_norm(
        self,
        df: DataFrame,
        srs_name: str = "norm",
        include_cols: Literal[False] = False,
    ) -> Series: ...

    @abstractmethod
    @overload
    def _df_norm(
        self,
        df: DataFrame,
        srs_name: str = "norm",
        include_cols: Literal[True] = False,
    ) -> DataFrame: ...

    @abstractmethod
    def _df_norm(
        self,
        df: DataFrame,
        srs_name: str = "norm",
        include_cols: bool = False,
    ) -> Series | DataFrame: ...

    @abstractmethod
    def _df_or(
        self,
        df: DataFrame,
        other: DataFrame | Sequence[float | int],
        axis: Literal["index", "columns"] = "index",
        index_cols: str | list[str] | None = None,
    ) -> DataFrame: ...

    @abstractmethod
    def _df_reindex(
        self,
        df: DataFrame,
        other: Sequence[Hashable] | DataFrame,
        new_index_cols: str | list[str],
        original_index_cols: str | list[str] | None = None,
    ) -> DataFrame: ...

    @abstractmethod
    def _df_rename_columns(
        self,
        df: DataFrame,
        old_columns: list[str],
        new_columns: list[str],
    ) -> DataFrame: ...

    @abstractmethod
    def _df_reset_index(
        self,
        df: DataFrame,
        index_cols: str | list[str] | None = None,
        drop: bool = False,
    ) -> DataFrame: ...

    @abstractmethod
    def _df_sample(
        self,
        df: DataFrame,
        n: int | None = None,
        frac: float | None = None,
        with_replacement: bool = False,
        shuffle: bool = False,
        seed: int | None = None,
    ) -> DataFrame: ...

    @abstractmethod
    def _df_set_index(
        self,
        df: DataFrame,
        index_name: str,
        new_index: Sequence[Hashable] | None = None,
    ) -> DataFrame: ...

    @abstractmethod
    def _df_with_columns(
        self,
        original_df: DataFrame,
        data: DataFrame
        | Series
        | Sequence[Sequence]
        | dict[str | Any]
        | Sequence[Any]
        | Any,
        new_columns: str | list[str] | None = None,
    ) -> DataFrame: ...

    @abstractmethod
    def _srs_constructor(
        self,
        data: Sequence[Any] | None = None,
        name: str | None = None,
        dtype: Any | None = None,
        index: Sequence[Any] | None = None,
    ) -> Series: ...

    @abstractmethod
    def _srs_contains(
        self,
        srs: Sequence[Any],
        values: Any | Sequence[Any],
    ) -> BoolSeries: ...

    @abstractmethod
    def _srs_range(self, name: str, start: int, end: int, step: int = 1) -> Series: ...

    @abstractmethod
    def _srs_to_df(self, srs: Series, index: Index | None = None) -> DataFrame: ...
