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
        backend classes (e.g. Polars implementations) to ensure consistent
        DataFrame manipulation across the mesa-frames package.
        
    LazyFrameMixin(ABC):
        A mixin class that defines an interface for lazy DataFrame operations.
        This mixin extends the DataFrameMixin with methods specific to lazy evaluation,
        allowing for the creation of query plans that are only executed when needed,
        which can improve performance with large datasets.

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
    The DataFrameMixin and LazyFrameMixin use Python's @abstractmethod decorator for their methods,
    ensuring that classes inheriting from them must implement these methods.

Attributes and methods of each mixin class are documented in their respective
docstrings.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection, Hashable, Iterator, Sequence
from copy import copy, deepcopy
from typing import Literal

from beartype import beartype
from typing import Any, Self, overload

from mesa_frames.types_ import (
    BoolSeries,
    DataFrame,
    Index,
    Mask,
    Series,
    DataFrameInput,
    LazyFrame,
    LazyIndex,
    BoolLF
)


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
        values: Collection[Any],
    ) -> BoolSeries: ...

    @abstractmethod
    def _df_constructor(
        self,
        data: DataFrameInput | None = None,
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
    def _df_index(self, df: DataFrame, index_name: str | Collection[str]) -> Index: ...

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
        include_cols: Literal[True] = True,
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
        other: Sequence[Hashable] | Index,
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
        index_name: str | Collection[str],
        new_index: Sequence[Hashable] | None = None,
    ) -> DataFrame: ...

    @abstractmethod
    def _df_with_columns(
        self,
        original_df: DataFrame,
        data: DataFrame
        | Series
        | Sequence[Sequence]
        | dict[str, Any]
        | Collection[Any]
        | Any,
        new_columns: str | list[str] | None = None,
    ) -> DataFrame: ...

    @abstractmethod
    def _srs_constructor(
        self,
        data: Collection[Any] | None = None,
        name: str | None = None,
        dtype: Any | None = None,
        index: Collection[Any] | None = None,
    ) -> Series: ...

    @abstractmethod
    def _srs_contains(
        self,
        srs: Collection[Any],
        values: Any | Collection[Any],
    ) -> BoolSeries: ...

    @abstractmethod
    def _srs_range(self, name: str, start: int, end: int, step: int = 1) -> Series: ...

    @abstractmethod
    def _srs_to_df(
        self, srs: Series, index: Collection[Any] | None = None
    ) -> DataFrame: ...


@beartype
class LazyFrameMixin(ABC):
    """A mixin class which defines an interface for lazy DataFrame operations.
    
    This mixin provides methods for creating and manipulating lazy DataFrames,
    which defer execution until results are actually needed. This can lead to
    significant performance improvements for complex operations on large datasets.
    
    Lazy evaluation allows the system to optimize query plans by combining
    multiple operations before execution, eliminating unnecessary intermediate
    results, and potentially improving memory usage and execution time.
    
    Classes implementing this mixin should provide concrete implementations of
    all abstract methods, ensuring consistent behavior across different backends.
    """
    
    def _lf_remove(self, lf: LazyFrame, mask: Mask, index_cols: str) -> LazyFrame:
        """Remove rows from a lazy DataFrame based on a mask.
        
        Parameters
        ----------
        lf : LazyFrame
            The lazy DataFrame to filter
        mask : Mask
            The mask indicating which rows to remove
        index_cols : str
            The column(s) to use as index
            
        Returns
        -------
        LazyFrame
            A new lazy DataFrame with the masked rows removed
        """
        return self._lf_get_masked_df(lf, index_cols, mask, negate=True)
    
    @abstractmethod
    def _lf_and(
        self,
        lf: LazyFrame,
        other: LazyFrame | Sequence[float | int],
        axis: Literal["index", "columns"] = "index",
        index_cols: str | list[str] | None = None,
    ) -> LazyFrame:
        """Perform bitwise AND operation on lazy DataFrames.
        
        Parameters
        ----------
        lf : LazyFrame
            The lazy DataFrame to operate on
        other : LazyFrame | Sequence[float | int]
            The other DataFrame or sequence to perform AND operation with
        axis : Literal["index", "columns"], optional
            Axis to perform operation along, by default "index"
        index_cols : str | list[str] | None, optional
            Column(s) to use as index, by default None
            
        Returns
        -------
        LazyFrame
            A new lazy DataFrame with the AND operation applied
        """
        ...
    
    @abstractmethod
    def _lf_add(
        self,
        lf: LazyFrame,
        other: LazyFrame | Sequence[float | int],
        axis: Literal["index", "columns"] = "index",
        index_cols: str | list[str] | None = None,
    ) -> LazyFrame:
        """Add values to a lazy DataFrame.
        
        Parameters
        ----------
        lf : LazyFrame
            The lazy DataFrame to add to
        other : LazyFrame | Sequence[float | int]
            The values to add
        axis : Literal["index", "columns"], optional
            Axis to perform addition along, by default "index"
        index_cols : str | list[str] | None, optional
            Column(s) to use as index, by default None
            
        Returns
        -------
        LazyFrame
            A new lazy DataFrame with the addition applied
        """
        ...
    
    @abstractmethod
    def _lf_all(
        self,
        lf: LazyFrame,
        name: str = "all",
        axis: str = "columns",
    ) -> LazyFrame:
        """Check if all values in the lazy DataFrame are True.
        
        Parameters
        ----------
        lf : LazyFrame
            The lazy DataFrame to check
        name : str, optional
            Name for the output series, by default "all"
        axis : str, optional
            Axis to check along, by default "columns"
            
        Returns
        -------
        Series
            A Series with True where all values along the axis are True
        """
        ...
    
    @abstractmethod
    def _lf_column_names(self, lf: LazyFrame) -> list[str]:
        """Get the column names of a lazy DataFrame.
        
        Parameters
        ----------
        lf : LazyFrame
            The lazy DataFrame to get column names from
            
        Returns
        -------
        list[str]
            List of column names
        """
        ...
    
    @abstractmethod
    def _lf_combine_first(
        self, 
        original_lf: LazyFrame, 
        new_lf: LazyFrame, 
        index_cols: str | list[str]
    ) -> LazyFrame:
        """Combine two lazy DataFrames, using values from new_lf where original_lf is null.
        
        Parameters
        ----------
        original_lf : LazyFrame
            The primary lazy DataFrame
        new_lf : LazyFrame
            The secondary lazy DataFrame with values to use when original_lf has nulls
        index_cols : str | list[str]
            Column(s) to use as index for combining
            
        Returns
        -------
        LazyFrame
            A new lazy DataFrame with combined values
        """
        ...
    
    @abstractmethod
    def _lf_concat(
        self,
        objs: Collection[LazyFrame],
        how: Literal["horizontal"] | Literal["vertical"] = "vertical",
        ignore_index: bool = False,
        index_cols: str | None = None,
    ) -> LazyFrame:
        """Concatenate multiple lazy DataFrames.
        
        Parameters
        ----------
        objs : Collection[LazyFrame]
            Collection of objects to concatenate
        how : Literal["horizontal"] | Literal["vertical"], optional
            Direction of concatenation, by default "vertical"
        ignore_index : bool, optional
            Whether to ignore the index, by default False
        index_cols : str | None, optional
            Column(s) to use as index, by default None
            
        Returns
        -------
        LazyFrame
            The concatenated result
        """
        ...
    
    @abstractmethod
    def _lf_contains(
        self,
        lf: LazyFrame,
        column: str,
        values: Collection[Any],
    ) -> BoolLF:
        """Check if values are contained in a column of the lazy DataFrame.
        
        Parameters
        ----------
        lf : LazyFrame
            The lazy DataFrame to check
        column : str
            The column to check
        values : Collection[Any]
            The values to check for
            
        Returns
        -------
        BoolLF
            A boolean lazy DataFrame indicating which values are contained
        """
        ...
    
    @abstractmethod
    def _lf_constructor(
        self,
        data: DataFrameInput | None = None,
        columns: list[str] | None = None,
        index: Index | None = None,
        index_cols: str | list[str] | None = None,
        dtypes: dict[str, Any] | None = None,
    ) -> LazyFrame:
        """Construct a new lazy DataFrame.
        
        Parameters
        ----------
        data : DataFrameInput | None, optional
            Data to initialize the lazy DataFrame with, by default None
        columns : list[str] | None, optional
            Column names for the lazy DataFrame, by default None
        index : Index | None, optional
            Index for the lazy DataFrame, by default None
        index_cols : str | list[str] | None, optional
            Name(s) of column(s) to use as index, by default None
        dtypes : dict[str, Any] | None, optional
            Data types for columns, by default None
            
        Returns
        -------
        LazyFrame
            A new lazy DataFrame instance
        """
        ...
    
    @abstractmethod
    def _lf_div(
        self,
        lf: LazyFrame,
        other: LazyFrame | Sequence[float | int],
        axis: Literal["index", "columns"] = "index",
        index_cols: str | list[str] | None = None,
    ) -> LazyFrame:
        """Divide a lazy DataFrame by another lazy DataFrame or sequence.
        
        Parameters
        ----------
        lf : LazyFrame
            The lazy DataFrame to divide
        other : LazyFrame | Sequence[float | int]
            The divisor
        axis : Literal["index", "columns"], optional
            Axis to perform division along, by default "index"
        index_cols : str | list[str] | None, optional
            Column(s) to use as index, by default None
            
        Returns
        -------
        LazyFrame
            A new lazy DataFrame with the division applied
        """
        ...
    
    @abstractmethod
    def _lf_drop_columns(
        self,
        lf: LazyFrame,
        columns: str | list[str],
    ) -> LazyFrame:
        """Drop columns from a lazy DataFrame.
        
        Parameters
        ----------
        lf : LazyFrame
            The lazy DataFrame to modify
        columns : str | list[str]
            The column(s) to drop
            
        Returns
        -------
        LazyFrame
            A new lazy DataFrame without the specified columns
        """
        ...
    
    @abstractmethod
    def _lf_drop_duplicates(
        self,
        lf: LazyFrame,
        subset: str | list[str] | None = None,
        keep: Literal["first", "last", False] = "first",
    ) -> LazyFrame:
        """Remove duplicate rows from a lazy DataFrame.
        
        Parameters
        ----------
        lf : LazyFrame
            The lazy DataFrame to process
        subset : str | list[str] | None, optional
            Column(s) to consider for identifying duplicates, by default None
        keep : Literal["first", "last", False], optional
            Which duplicates to keep, by default "first"
            
        Returns
        -------
        LazyFrame
            A new lazy DataFrame with duplicates removed
        """
        ...
    
    @abstractmethod
    def _lf_ge(
        self,
        lf: LazyFrame,
        other: LazyFrame | Sequence[float | int],
        axis: Literal["index", "columns"] = "index",
        index_cols: str | list[str] | None = None,
    ) -> LazyFrame:
        """Compare if lazy DataFrame values are greater than or equal to another DataFrame or sequence.
        
        Parameters
        ----------
        lf : LazyFrame
            The lazy DataFrame to compare
        other : LazyFrame | Sequence[float | int]
            The values to compare against
        axis : Literal["index", "columns"], optional
            Axis to compare along, by default "index"
        index_cols : str | list[str] | None, optional
            Column(s) to use as index, by default None
            
        Returns
        -------
        LazyFrame
            A new lazy DataFrame with boolean values from the comparison
        """
        ...
    
    @abstractmethod
    def _lf_get_bool_mask(
        self,
        lf: LazyFrame,
        index_cols: str | list[str] | None = None,
        mask: Mask | None = None,
        negate: bool = False,
    ) -> BoolLF:
        """Get a boolean mask for filtering a lazy DataFrame.
        
        Parameters
        ----------
        lf : LazyFrame
            The lazy DataFrame to create a mask for
        index_cols : str | list[str] | None, optional
            Column(s) to use as index, by default None
        mask : Mask | None, optional
            The mask to convert to a boolean series, by default None
        negate : bool, optional
            Whether to negate the mask, by default False
            
        Returns
        -------
        BoolLF
            A boolean lazy DataFrame indicating which rows to keep
        """
        ...
    
    @abstractmethod
    def _lf_get_masked_df(
        self,
        lf: LazyFrame,
        index_cols: str | list[str] | None,
        mask: Mask | None = None,
        columns: str | list[str] | None = None,
        negate: bool = False,
    ) -> LazyFrame:
        """Get a filtered lazy DataFrame based on a mask.
        
        Parameters
        ----------
        lf : LazyFrame
            The lazy DataFrame to filter
        index_cols : str | list[str] | None, optional
            Column(s) to use as index, by default None
        mask : Mask | None, optional
            The mask to filter by, by default None
        columns : str | list[str] | None, optional
            Column(s) to select in the result, by default None
        negate : bool, optional
            Whether to negate the mask, by default False
            
        Returns
        -------
        LazyFrame
            A new filtered lazy DataFrame
        """
        ...
    
    @abstractmethod
    def _lf_groupby_cumcount(
        self, 
        lf: LazyFrame, 
        by: str | list[str], 
        name: str = "cum_count"
    ) -> LazyFrame:
        """Compute cumulative count for each group in a lazy DataFrame.
        
        Parameters
        ----------
        lf : LazyFrame
            The lazy DataFrame to group
        by : str | list[str]
            Column(s) to group by
        name : str, optional
            Name for the output LazyFrame, by default "cum_count"
            
        Returns
        -------
        LazyFrame
            A LazyFrame with cumulative counts for each group
        """
        ...
    
    @abstractmethod
    def _lf_index(
        self, 
        lf: LazyFrame, 
        index_name: str | Collection[str]
    ) -> LazyIndex:
        """Get the index of a lazy DataFrame.
        
        Parameters
        ----------
        lf : LazyFrame
            The lazy DataFrame to get the index from
        index_name : str | Collection[str]
            Name(s) of column(s) to use as index
            
        Returns
        -------
        Index
            The index of the lazy DataFrame
        """
        ...
    
    @abstractmethod
    def _lf_iterator(self, lf: LazyFrame) -> Iterator[dict[str, Any]]:
        """Get an iterator over rows of a lazy DataFrame.
        
        Parameters
        ----------
        lf : LazyFrame
            The lazy DataFrame to iterate over
            
        Returns
        -------
        Iterator[dict[str, Any]]
            An iterator yielding rows as dictionaries
        """
        ...
    
    @abstractmethod
    def _lf_join(
        self,
        left: LazyFrame,
        right: LazyFrame,
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
    ) -> LazyFrame:
        """Join two lazy DataFrames.
        
        Parameters
        ----------
        left : LazyFrame
            The left lazy DataFrame
        right : LazyFrame
            The right lazy DataFrame
        index_cols : str | list[str] | None, optional
            Column(s) to use as index, by default None
        on : str | list[str] | None, optional
            Column(s) to join on, by default None
        left_on : str | list[str] | None, optional
            Column(s) from left DataFrame to join on, by default None
        right_on : str | list[str] | None, optional
            Column(s) from right DataFrame to join on, by default None
        how : Literal["left", "right", "inner", "outer", "cross"], optional
            Type of join to perform, by default "left"
        suffix : str, optional
            Suffix to add to overlapping column names, by default "_right"
            
        Returns
        -------
        LazyFrame
            A new lazy DataFrame resulting from the join
        """
        ...
    
    @abstractmethod
    def _lf_lt(
        self,
        lf: LazyFrame,
        other: LazyFrame | Sequence[float | int],
        axis: Literal["index", "columns"] = "index",
        index_cols: str | list[str] | None = None,
    ) -> LazyFrame:
        """Compare if lazy DataFrame values are less than another DataFrame or sequence.
        
        Parameters
        ----------
        lf : LazyFrame
            The lazy DataFrame to compare
        other : LazyFrame | Sequence[float | int]
            The values to compare against
        axis : Literal["index", "columns"], optional
            Axis to compare along, by default "index"
        index_cols : str | list[str] | None, optional
            Column(s) to use as index, by default None
            
        Returns
        -------
        LazyFrame
            A new lazy DataFrame with boolean values from the comparison
        """
        ...
    
    @abstractmethod
    def _lf_mod(
        self,
        lf: LazyFrame,
        other: LazyFrame | Sequence[float | int],
        axis: Literal["index", "columns"] = "index",
        index_cols: str | list[str] | None = None,
    ) -> LazyFrame:
        """Compute element-wise modulo of lazy DataFrame values.
        
        Parameters
        ----------
        lf : LazyFrame
            The lazy DataFrame to apply modulo to
        other : LazyFrame | Sequence[float | int]
            The values to compute modulo with
        axis : Literal["index", "columns"], optional
            Axis to apply operation along, by default "index"
        index_cols : str | list[str] | None, optional
            Column(s) to use as index, by default None
            
        Returns
        -------
        LazyFrame
            A new lazy DataFrame with modulo results
        """
        ...
    
    @abstractmethod
    def _lf_mul(
        self,
        lf: LazyFrame,
        other: LazyFrame | Sequence[float | int],
        axis: Literal["index", "columns"] = "index",
        index_cols: str | list[str] | None = None,
    ) -> LazyFrame:
        """Multiply lazy DataFrame values by another DataFrame or sequence.
        
        Parameters
        ----------
        lf : LazyFrame
            The lazy DataFrame to multiply
        other : LazyFrame | Sequence[float | int]
            The values to multiply by
        axis : Literal["index", "columns"], optional
            Axis to apply multiplication along, by default "index"
        index_cols : str | list[str] | None, optional
            Column(s) to use as index, by default None
            
        Returns
        -------
        LazyFrame
            A new lazy DataFrame with multiplication results
        """
        ...

    @abstractmethod
    def _lf_norm(
        self,
        lf: LazyFrame,
        srs_name: str = "norm",
        include_cols: bool = False,
    ) -> LazyFrame:
        """Compute the Euclidean norm of each row in a lazy DataFrame.
        
        Parameters
        ----------
        lf : LazyFrame
            The lazy DataFrame to compute norms for
        srs_name : str, optional
            Name for the output series, by default "norm"
        include_cols : bool, optional
            Whether to include the norm as a column in the DataFrame, by default False
            
        Returns
        -------
        LazyFrame
            If include_cols is False, returns a LazyFrame only with the norm series.
            If include_cols is True, returns the input LazyFrame with an added norm column.
        """
        ...
    
    @abstractmethod
    def _lf_or(
        self,
        lf: LazyFrame,
        other: LazyFrame | Sequence[float | int],
        axis: Literal["index", "columns"] = "index",
        index_cols: str | list[str] | None = None,
    ) -> LazyFrame:
        """Perform bitwise OR operation on lazy DataFrames.
        
        Parameters
        ----------
        lf : LazyFrame
            The lazy DataFrame to operate on
        other : LazyFrame | Sequence[float | int]
            The other DataFrame or sequence to perform OR operation with
        axis : Literal["index", "columns"], optional
            Axis to perform operation along, by default "index"
        index_cols : str | list[str] | None, optional
            Column(s) to use as index, by default None
            
        Returns
        -------
        LazyFrame
            A new lazy DataFrame with the OR operation applied
        """
        ...
    
    @abstractmethod
    def _lf_reindex(
        self,
        lf: LazyFrame,
        other: Sequence[Hashable] | Index,
        new_index_cols: str | list[str],
        original_index_cols: str | list[str] | None = None,
    ) -> LazyFrame:
        """Reindex a lazy DataFrame.
        
        Parameters
        ----------
        lf : LazyFrame
            The lazy DataFrame to reindex
        other : Sequence[Hashable] | Index
            The new index values
        new_index_cols : str | list[str]
            Name(s) of column(s) to use as the new index
        original_index_cols : str | list[str] | None, optional
            Name(s) of column(s) currently used as index, by default None
            
        Returns
        -------
        LazyFrame
            A new lazy DataFrame with the new index
        """
        ...
    
    @abstractmethod
    def _lf_rename_columns(
        self,
        lf: LazyFrame,
        old_columns: list[str],
        new_columns: list[str],
    ) -> LazyFrame:
        """Rename columns in a lazy DataFrame.
        
        Parameters
        ----------
        lf : LazyFrame
            The lazy DataFrame to modify
        old_columns : list[str]
            Current column names
        new_columns : list[str]
            New column names
            
        Returns
        -------
        LazyFrame
            A new lazy DataFrame with renamed columns
        """
        ...
    
    @abstractmethod
    def _lf_reset_index(
        self,
        lf: LazyFrame,
        index_cols: str | list[str] | None = None,
        drop: bool = False,
    ) -> LazyFrame:
        """Reset the index of a lazy DataFrame.
        
        Parameters
        ----------
        lf : LazyFrame
            The lazy DataFrame to modify
        index_cols : str | list[str] | None, optional
            Column(s) to reset, by default None
        drop : bool, optional
            Whether to drop the index columns, by default False
            
        Returns
        -------
        LazyFrame
            A new lazy DataFrame with reset index
        """
        ...
    
    @abstractmethod
    def _lf_sample(
        self,
        lf: LazyFrame,
        n: int | None = None,
        frac: float | None = None,
        with_replacement: bool = False,
        shuffle: bool = False,
        seed: int | None = None,
    ) -> LazyFrame:
        """Sample rows from a lazy DataFrame.
        
        Parameters
        ----------
        lf : LazyFrame
            The lazy DataFrame to sample from
        n : int | None, optional
            Number of samples to draw, by default None
        frac : float | None, optional
            Fraction of rows to sample, by default None
        with_replacement : bool, optional
            Sample with replacement, by default False
        shuffle : bool, optional
            Shuffle the result, by default False
        seed : int | None, optional
            Random seed, by default None
            
        Returns
        -------
        LazyFrame
            A new lazy DataFrame with sampled rows
        """
        ...
    
    @abstractmethod
    def _lf_set_index(
        self,
        lf: LazyFrame,
        index_name: str | Collection[str],
        new_index: Collection[Hashable] | None = None,
    ) -> LazyFrame:
        """Set the index of a lazy DataFrame.
        
        Parameters
        ----------
        lf : LazyFrame
            The lazy DataFrame to modify
        index_name : str | Collection[str]
            Name(s) of column(s) to use as index
        new_index : Collection[Hashable] | None, optional
            New index values, by default None
            
        Returns
        -------
        LazyFrame
            A new lazy DataFrame with the new index
        """
        ...
    
    @abstractmethod
    def _lf_with_columns(
        self,
        original_lf: LazyFrame,
        data: LazyFrame
        | Series
        | Sequence[Sequence]
        | dict[str, Any]
        | Collection[Any]
        | Any,
        new_columns: str | list[str] | None = None,
    ) -> LazyFrame:
        """Add or modify columns in a lazy DataFrame.
        
        Parameters
        ----------
        original_lf : LazyFrame
            The lazy DataFrame to modify
        data : LazyFrame | Series | Sequence[Sequence] | dict[str, Any] | Collection[Any] | Any
            Data for the new or modified columns
        new_columns : str | list[str] | None, optional
            Names for the new columns, by default None
            
        Returns
        -------
        LazyFrame
            A new lazy DataFrame with added or modified columns
        """
        ...