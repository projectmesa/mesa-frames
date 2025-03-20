import numpy as np
import polars as pl
import pytest
import typeguard as tg

from mesa_frames.concrete.polars.mixin import PolarsMixin


@tg.typechecked
class TestPolarsMixin:
    @pytest.fixture
    def mixin(self):
        return PolarsMixin()

    @pytest.fixture
    def df_0(self):
        return pl.DataFrame(
            {
                "unique_id": ["x", "y", "z"],
                "A": [1, 2, 3],
                "B": ["a", "b", "c"],
                "C": [True, False, True],
                "D": [1, 2, 3],
            },
        )

    @pytest.fixture
    def df_1(self):
        return pl.DataFrame(
            {
                "unique_id": ["z", "a", "b"],
                "A": [4, 5, 6],
                "B": ["d", "e", "f"],
                "C": [False, True, False],
                "E": [1, 2, 3],
            },
        )

    def test_df_add(self, mixin: PolarsMixin, df_0: pl.DataFrame, df_1: pl.DataFrame):
        # Test adding a DataFrame and a sequence element-wise along the rows (axis='index')
        result = mixin._df_add(df_0[["A", "D"]], df_1["A"], axis="index")
        assert isinstance(result, pl.DataFrame)
        assert result["A"].to_list() == [5, 7, 9]
        assert result["D"].to_list() == [5, 7, 9]

        # Test adding a DataFrame and a sequence element-wise along the column (axis='columns')
        result = mixin._df_add(df_0[["A", "D"]], [1, 2], axis="columns")
        assert isinstance(result, pl.DataFrame)
        assert result["A"].to_list() == [2, 3, 4]
        assert result["D"].to_list() == [3, 4, 5]

        # Test adding DataFrames with index-column alignment
        df_1 = df_1.with_columns(D=pl.col("E"))
        result = mixin._df_add(
            df_0[["unique_id", "A", "D"]],
            df_1[["unique_id", "A", "D"]],
            axis="index",
            index_cols="unique_id",
        )
        assert isinstance(result, pl.DataFrame)
        assert result["A"].to_list() == [None, None, 7]
        assert result["D"].to_list() == [None, None, 4]

    def test_df_all(self, mixin: PolarsMixin):
        df = pl.DataFrame(
            {
                "A": [True, False, True],
                "B": [True, True, True],
            }
        )

        # Test with axis='columns'
        result = mixin._df_all(df["A", "B"], axis="columns")
        assert isinstance(result, pl.Series)
        assert result.name == "all"
        assert result.to_list() == [True, False, True]

        # Test with axis='index'
        result = mixin._df_all(df["A", "B"], axis="index")
        assert isinstance(result, pl.Series)
        assert result.name == "all"
        assert result.to_list() == [False, True]

    def test_df_and(self, mixin: PolarsMixin, df_0: pl.DataFrame, df_1: pl.DataFrame):
        # Test comparing the DataFrame with a sequence element-wise along the rows (axis='index')
        df_0 = df_0.with_columns(F=pl.Series([True, True, False]))
        df_1 = df_1.with_columns(F=pl.Series([False, False, True]))
        result = mixin._df_and(df_0[["C", "F"]], df_1["F"], axis="index")
        assert isinstance(result, pl.DataFrame)
        assert result["C"].to_list() == [False, False, True]
        assert result["F"].to_list() == [False, False, False]

        # Test comparing the DataFrame with a sequence element-wise along the columns (axis='columns')
        result = mixin._df_and(df_0[["C", "F"]], [True, False], axis="columns")
        assert isinstance(result, pl.DataFrame)
        assert result["C"].to_list() == [True, False, True]
        assert result["F"].to_list() == [False, False, False]

        # Test comparing DataFrames with index-column alignment
        result = mixin._df_and(
            df_0[["unique_id", "C", "F"]],
            df_1[["unique_id", "C", "F"]],
            axis="index",
            index_cols="unique_id",
        )
        assert isinstance(result, pl.DataFrame)
        assert result["C"].to_list() == [None, False, False]
        assert result["F"].to_list() == [None, None, False]

    def test_df_column_names(self, mixin: PolarsMixin, df_0: pl.DataFrame):
        cols = mixin._df_column_names(df_0)
        assert isinstance(cols, list)
        assert all(isinstance(c, str) for c in cols)
        assert set(mixin._df_column_names(df_0)) == {"unique_id", "A", "B", "C", "D"}

    def test_df_combine_first(
        self, mixin: PolarsMixin, df_0: pl.DataFrame, df_1: pl.DataFrame
    ):
        # Test with df_0 and df_1
        result = mixin._df_combine_first(df_0, df_1, "unique_id")
        result = result.sort("A")
        assert isinstance(result, pl.DataFrame)
        assert set(result.columns) == {"unique_id", "A", "B", "C", "D", "E"}
        assert result["unique_id"].to_list() == ["x", "y", "z", "a", "b"]
        assert result["A"].to_list() == [1, 2, 3, 5, 6]
        assert result["B"].to_list() == ["a", "b", "c", "e", "f"]
        assert result["C"].to_list() == [True, False, True, True, False]
        assert result["D"].to_list() == [1, 2, 3, None, None]
        assert result["E"].to_list() == [None, None, 1, 2, 3]

        # Test with df_1 and df_0
        result = mixin._df_combine_first(df_1, df_0, "unique_id")
        result = result.sort("E", nulls_last=True)
        assert isinstance(result, pl.DataFrame)
        assert set(result.columns) == {"unique_id", "A", "B", "C", "D", "E"}
        assert result["unique_id"].to_list() == ["z", "a", "b", "x", "y"]
        assert result["A"].to_list() == [4, 5, 6, 1, 2]
        assert result["B"].to_list() == ["d", "e", "f", "a", "b"]
        assert result["C"].to_list() == [False, True, False, True, False]
        assert result["D"].to_list() == [3, None, None, 1, 2]
        assert result["E"].to_list() == [1, 2, 3, None, None]

    def test_df_concat(
        self, mixin: PolarsMixin, df_0: pl.DataFrame, df_1: pl.DataFrame
    ):
        ### Test vertical concatenation
        ## With DataFrames
        for ignore_index in [False, True]:
            vertical = mixin._df_concat(
                [df_0, df_1], how="vertical", ignore_index=ignore_index
            )
            assert isinstance(vertical, pl.DataFrame)
            assert vertical.columns == ["unique_id", "A", "B", "C", "D", "E"]
            assert len(vertical) == 6
            assert vertical["unique_id"].to_list() == ["x", "y", "z", "z", "a", "b"]
            assert vertical["A"].to_list() == [1, 2, 3, 4, 5, 6]
            assert vertical["B"].to_list() == ["a", "b", "c", "d", "e", "f"]
            assert vertical["C"].to_list() == [True, False, True, False, True, False]
            assert vertical["D"].to_list() == [1, 2, 3, None, None, None]
            assert vertical["E"].to_list() == [None, None, None, 1, 2, 3]

        ## With Series
        for ignore_index in [True, False]:
            vertical = mixin._df_concat(
                [df_0["A"], df_1["A"]], how="vertical", ignore_index=ignore_index
            )
            assert isinstance(vertical, pl.Series)
            assert len(vertical) == 6
            assert vertical.to_list() == [1, 2, 3, 4, 5, 6]
            assert vertical.name == "A"

        ## Test horizontal concatenation
        ## With DataFrames
        # Error With same column names
        with pytest.raises(pl.exceptions.DuplicateError):
            mixin._df_concat([df_0, df_1], how="horizontal")
        # With ignore_index = False
        df_1 = df_1.rename(lambda c: f"{c}_1")
        horizontal = mixin._df_concat([df_0, df_1], how="horizontal")
        assert isinstance(horizontal, pl.DataFrame)
        assert horizontal.columns == [
            "unique_id",
            "A",
            "B",
            "C",
            "D",
            "unique_id_1",
            "A_1",
            "B_1",
            "C_1",
            "E_1",
        ]
        assert len(horizontal) == 3
        assert horizontal["unique_id"].to_list() == ["x", "y", "z"]
        assert horizontal["A"].to_list() == [1, 2, 3]
        assert horizontal["B"].to_list() == ["a", "b", "c"]
        assert horizontal["C"].to_list() == [True, False, True]
        assert horizontal["D"].to_list() == [1, 2, 3]
        assert horizontal["unique_id_1"].to_list() == ["z", "a", "b"]
        assert horizontal["A_1"].to_list() == [4, 5, 6]
        assert horizontal["B_1"].to_list() == ["d", "e", "f"]
        assert horizontal["C_1"].to_list() == [False, True, False]
        assert horizontal["E_1"].to_list() == [1, 2, 3]

        # With ignore_index = True
        horizontal_ignore_index = mixin._df_concat(
            [df_0, df_1],
            how="horizontal",
            ignore_index=True,
        )
        assert isinstance(horizontal_ignore_index, pl.DataFrame)
        assert horizontal_ignore_index.columns == [
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
        ]
        assert len(horizontal_ignore_index) == 3
        assert horizontal_ignore_index["0"].to_list() == ["x", "y", "z"]
        assert horizontal_ignore_index["1"].to_list() == [1, 2, 3]
        assert horizontal_ignore_index["2"].to_list() == ["a", "b", "c"]
        assert horizontal_ignore_index["3"].to_list() == [True, False, True]
        assert horizontal_ignore_index["4"].to_list() == [1, 2, 3]
        assert horizontal_ignore_index["5"].to_list() == ["z", "a", "b"]
        assert horizontal_ignore_index["6"].to_list() == [4, 5, 6]
        assert horizontal_ignore_index["7"].to_list() == ["d", "e", "f"]
        assert horizontal_ignore_index["8"].to_list() == [False, True, False]
        assert horizontal_ignore_index["9"].to_list() == [1, 2, 3]

        ## With Series
        # With ignore_index = False
        horizontal = mixin._df_concat(
            [df_0["A"], df_1["B_1"]], how="horizontal", ignore_index=False
        )
        assert isinstance(horizontal, pl.DataFrame)
        assert horizontal.columns == ["A", "B_1"]
        assert len(horizontal) == 3
        assert horizontal["A"].to_list() == [1, 2, 3]
        assert horizontal["B_1"].to_list() == ["d", "e", "f"]

        # With ignore_index = True
        horizontal = mixin._df_concat(
            [df_0["A"], df_1["B_1"]], how="horizontal", ignore_index=True
        )
        assert isinstance(horizontal, pl.DataFrame)
        assert horizontal.columns == ["0", "1"]
        assert len(horizontal) == 3
        assert horizontal["0"].to_list() == [1, 2, 3]
        assert horizontal["1"].to_list() == ["d", "e", "f"]

    def test_df_constructor(self, mixin: PolarsMixin):
        # Test with dictionary
        data = {"num": [1, 2, 3], "letter": ["a", "b", "c"]}
        df = mixin._df_constructor(data)
        assert isinstance(df, pl.DataFrame)

        assert list(df.columns) == ["num", "letter"]
        assert df["num"].to_list() == [1, 2, 3]
        assert df["letter"].to_list() == ["a", "b", "c"]

        # Test with list of lists
        data = [[1, "a"], [2, "b"], [3, "c"]]
        df = mixin._df_constructor(
            data, columns=["num", "letter"], dtypes={"num": "int64"}
        )
        assert isinstance(df, pl.DataFrame)
        assert list(df.columns) == ["num", "letter"]
        assert df["num"].dtype == pl.Int64
        assert df["num"].to_list() == [1, 2, 3]
        assert df["letter"].to_list() == ["a", "b", "c"]

        # Test with index > 1 and 1 value
        data = {"a": 5}
        df = mixin._df_constructor(
            data, index=pl.int_range(5, eager=True), index_cols="index"
        )
        assert isinstance(df, pl.DataFrame)
        assert list(df.columns) == ["index", "a"]
        assert df["a"].to_list() == [5, 5, 5, 5, 5]
        assert df["index"].to_list() == [0, 1, 2, 3, 4]

    def test_df_contains(self, mixin: PolarsMixin, df_0: pl.DataFrame):
        # Test with list
        result = mixin._df_contains(df_0, "A", [5, 2, 3])
        assert isinstance(result, pl.Series)
        assert result.name == "contains"
        assert result.to_list() == [False, True, True]

    def test_df_div(self, mixin: PolarsMixin, df_0: pl.DataFrame, df_1: pl.DataFrame):
        # Test dividing the DataFrame by a sequence element-wise along the rows (axis='index')
        result = mixin._df_div(df_0[["A", "D"]], df_1["A"], axis="index")
        assert isinstance(result, pl.DataFrame)
        assert result["A"].to_list() == [0.25, 0.4, 0.5]
        assert result["D"].to_list() == [0.25, 0.4, 0.5]

        # Test dividing the DataFrame by a sequence element-wise along the columns (axis='columns')
        result = mixin._df_div(df_0[["A", "D"]], [1, 2], axis="columns")
        assert isinstance(result, pl.DataFrame)
        assert result["A"].to_list() == [1, 2, 3]
        assert result["D"].to_list() == [0.5, 1, 1.5]

        # Test dividing DataFrames with index-column alignment
        df_1 = df_1.with_columns(D=pl.col("E"))
        result = mixin._df_div(
            df_0[["unique_id", "A", "D"]],
            df_1[["unique_id", "A", "D"]],
            axis="index",
            index_cols="unique_id",
        )
        assert isinstance(result, pl.DataFrame)
        assert result["A"].to_list() == [None, None, 0.75]
        assert result["D"].to_list() == [None, None, 3]

    def test_df_drop_columns(self, mixin: PolarsMixin, df_0: pl.DataFrame):
        # Test with str
        dropped = mixin._df_drop_columns(df_0, "A")
        assert isinstance(dropped, pl.DataFrame)
        assert dropped.columns == ["unique_id", "B", "C", "D"]
        # Test with list
        dropped = mixin._df_drop_columns(df_0, ["A", "C"])
        assert dropped.columns == ["unique_id", "B", "D"]

    def test_df_drop_duplicates(self, mixin: PolarsMixin, df_0: pl.DataFrame):
        new_df = pl.concat([df_0, df_0], how="vertical")
        assert len(new_df) == 6

        # Test with all columns
        dropped = mixin._df_drop_duplicates(new_df)
        assert isinstance(dropped, pl.DataFrame)
        assert len(dropped) == 3
        assert dropped.columns == ["unique_id", "A", "B", "C", "D"]

        # Test with subset (str)
        other_df = pl.DataFrame(
            {
                "unique_id": ["x", "y", "z"],
                "A": [1, 2, 3],
                "B": ["d", "e", "f"],
                "C": [True, True, False],
                "D": [1, 2, 3],
            },
        )
        new_df = pl.concat([df_0, other_df], how="vertical")
        dropped = mixin._df_drop_duplicates(new_df, subset="unique_id")
        assert isinstance(dropped, pl.DataFrame)
        assert len(dropped) == 3

        # Test with subset (list)
        dropped = mixin._df_drop_duplicates(new_df, subset=["A", "C"])
        assert isinstance(dropped, pl.DataFrame)
        assert len(dropped) == 5
        assert dropped.columns == ["unique_id", "A", "B", "C", "D"]
        assert dropped["B"].to_list() == ["a", "b", "c", "e", "f"]

        # Test with subset (list) and keep='last'
        dropped = mixin._df_drop_duplicates(new_df, subset=["A", "C"], keep="last")
        assert isinstance(dropped, pl.DataFrame)
        assert len(dropped) == 5
        assert dropped.columns == ["unique_id", "A", "B", "C", "D"]
        assert dropped["B"].to_list() == ["d", "b", "c", "e", "f"]

        # Test with subset (list) and keep=False
        dropped = mixin._df_drop_duplicates(new_df, subset=["A", "C"], keep=False)
        assert isinstance(dropped, pl.DataFrame)
        assert len(dropped) == 4
        assert dropped.columns == ["unique_id", "A", "B", "C", "D"]
        assert dropped["B"].to_list() == ["b", "c", "e", "f"]

    def test_df_ge(self, mixin: PolarsMixin, df_0: pl.DataFrame, df_1: pl.DataFrame):
        # Test comparing the DataFrame with a sequence element-wise along the rows (axis='index')
        result = mixin._df_ge(df_0[["A", "D"]], df_1["A"], axis="index")
        assert isinstance(result, pl.DataFrame)
        assert result["A"].to_list() == [False, False, False]
        assert result["D"].to_list() == [False, False, False]

        # Test comparing the DataFrame with a sequence element-wise along the columns (axis='columns')
        result = mixin._df_ge(df_0[["A", "D"]], [1, 2], axis="columns")
        assert isinstance(result, pl.DataFrame)
        assert result["A"].to_list() == [True, True, True]
        assert result["D"].to_list() == [False, True, True]

        # Test comparing DataFrames with index-column alignment
        df_1 = df_1.with_columns(D=pl.col("E"))
        result = mixin._df_ge(
            df_0[["unique_id", "A", "D"]],
            df_1[["unique_id", "A", "D"]],
            axis="index",
            index_cols="unique_id",
        )
        assert isinstance(result, pl.DataFrame)
        assert result["A"].to_list() == [None, None, False]
        assert result["D"].to_list() == [None, None, True]

    def test_df_get_bool_mask(self, mixin: PolarsMixin, df_0: pl.DataFrame):
        # Test with pl.Series[bool]
        mask = mixin._df_get_bool_mask(df_0, "A", pl.Series([True, False, True]))
        assert mask.to_list() == [True, False, True]

        # Test with DataFrame
        mask_df = pl.DataFrame({"A": [1, 3]})
        mask = mixin._df_get_bool_mask(df_0, "A", mask_df)
        assert mask.to_list() == [True, False, True]

        # Test with single value
        mask = mixin._df_get_bool_mask(df_0, "A", 1)
        assert mask.to_list() == [True, False, False]

        # Test with list of values
        mask = mixin._df_get_bool_mask(df_0, "A", [1, 3])
        assert mask.to_list() == [True, False, True]

        # Test with negate=True
        mask = mixin._df_get_bool_mask(df_0, "A", [1, 3], negate=True)
        assert mask.to_list() == [False, True, False]

    def test_df_get_masked_df(self, mixin: PolarsMixin, df_0: pl.DataFrame):
        # Test with pl.Series[bool]
        masked_df = mixin._df_get_masked_df(df_0, "A", pl.Series([True, False, True]))
        assert masked_df["A"].to_list() == [1, 3]
        assert masked_df["unique_id"].to_list() == ["x", "z"]

        # Test with DataFrame
        mask_df = pl.DataFrame({"A": [1, 3]})
        masked_df = mixin._df_get_masked_df(df_0, "A", mask_df)
        assert masked_df["A"].to_list() == [1, 3]
        assert masked_df["unique_id"].to_list() == ["x", "z"]

        # Test with single value
        masked_df = mixin._df_get_masked_df(df_0, "A", 1)
        assert masked_df["A"].to_list() == [1]
        assert masked_df["unique_id"].to_list() == ["x"]

        # Test with list of values
        masked_df = mixin._df_get_masked_df(df_0, "A", [1, 3])
        assert masked_df["A"].to_list() == [1, 3]
        assert masked_df["unique_id"].to_list() == ["x", "z"]

        # Test with columns
        masked_df = mixin._df_get_masked_df(df_0, "A", [1, 3], columns=["B"])
        assert list(masked_df.columns) == ["B"]
        assert masked_df["B"].to_list() == ["a", "c"]

        # Test with negate=True
        masked = mixin._df_get_masked_df(df_0, "A", [1, 3], negate=True)
        assert len(masked) == 1

    def test_df_groupby_cumcount(self, df_0: pl.DataFrame, mixin: PolarsMixin):
        result = mixin._df_groupby_cumcount(df_0, "C")
        assert result.to_list() == [1, 1, 2]

    def test_df_index(self, mixin: PolarsMixin, df_0: pl.DataFrame):
        index = mixin._df_index(df_0, "unique_id")
        assert isinstance(index, pl.Series)
        assert index.to_list() == ["x", "y", "z"]

    def test_df_iterator(self, mixin: PolarsMixin, df_0: pl.DataFrame):
        iterator = mixin._df_iterator(df_0)
        first_item = next(iterator)
        assert first_item == {"unique_id": "x", "A": 1, "B": "a", "C": True, "D": 1}

    def test_df_join(self, mixin: PolarsMixin):
        left = pl.DataFrame({"A": [1, 2], "B": ["a", "b"]})
        right = pl.DataFrame({"A": [1, 3], "C": ["x", "y"]})

        # Test with 'on' (left join)
        joined = mixin._df_join(left, right, on="A")
        assert set(joined.columns) == {"A", "B", "C"}
        assert joined["A"].to_list() == [1, 2]

        # Test with 'left_on' and 'right_on' (left join)
        right_1 = pl.DataFrame({"D": [1, 2], "C": ["x", "y"]})
        joined = mixin._df_join(left, right_1, left_on="A", right_on="D")
        assert set(joined.columns) == {"A", "B", "C"}
        assert joined["A"].to_list() == [1, 2]

        # Test with 'right' join
        joined = mixin._df_join(left, right, on="A", how="right")
        assert set(joined.columns) == {"A", "B", "C"}
        assert joined["A"].to_list() == [1, 3]

        # Test with 'inner' join
        joined = mixin._df_join(left, right, on="A", how="inner")
        assert set(joined.columns) == {"A", "B", "C"}
        assert joined["A"].to_list() == [1]

        # Test with 'outer' join
        joined = mixin._df_join(left, right, on="A", how="outer")
        assert set(joined.columns) == {"A", "B", "A_right", "C"}
        assert joined["A"].to_list() == [1, None, 2]
        assert joined["A_right"].to_list() == [1, 3, None]

        # Test with 'cross' join
        joined = mixin._df_join(left, right, how="cross")
        assert set(joined.columns) == {"A", "B", "A_right", "C"}
        assert len(joined) == 4
        assert joined.row(0) == (1, "a", 1, "x")
        assert joined.row(1) == (1, "a", 3, "y")
        assert joined.row(2) == (2, "b", 1, "x")
        assert joined.row(3) == (2, "b", 3, "y")

        # Test with different 'suffix'
        joined = mixin._df_join(left, right, suffix="_r", how="cross")
        assert set(joined.columns) == {"A", "B", "A_r", "C"}
        assert len(joined) == 4
        assert joined.row(0) == (1, "a", 1, "x")
        assert joined.row(1) == (1, "a", 3, "y")
        assert joined.row(2) == (2, "b", 1, "x")
        assert joined.row(3) == (2, "b", 3, "y")

    def test_df_lt(self, mixin: PolarsMixin, df_0: pl.DataFrame, df_1: pl.DataFrame):
        # Test comparing the DataFrame with a sequence element-wise along the rows (axis='index')
        result = mixin._df_lt(df_0[["A", "D"]], df_1["A"], axis="index")
        assert isinstance(result, pl.DataFrame)
        assert result["A"].to_list() == [True, True, True]
        assert result["D"].to_list() == [True, True, True]

        # Test comparing the DataFrame with a sequence element-wise along the columns (axis='columns')
        result = mixin._df_lt(df_0[["A", "D"]], [2, 3], axis="columns")
        assert isinstance(result, pl.DataFrame)
        assert result["A"].to_list() == [True, False, False]
        assert result["D"].to_list() == [True, True, False]

        # Test comparing DataFrames with index-column alignment
        df_1 = df_1.with_columns(D=pl.col("E"))
        result = mixin._df_lt(
            df_0[["unique_id", "A", "D"]],
            df_1[["unique_id", "A", "D"]],
            axis="index",
            index_cols="unique_id",
        )
        assert isinstance(result, pl.DataFrame)
        assert result["A"].to_list() == [None, None, True]
        assert result["D"].to_list() == [None, None, False]

    def test_df_mod(self, mixin: PolarsMixin, df_0: pl.DataFrame, df_1: pl.DataFrame):
        # Test taking the modulo of the DataFrame by a sequence element-wise along the rows (axis='index')
        result = mixin._df_mod(df_0[["A", "D"]], df_1["A"], axis="index")
        assert isinstance(result, pl.DataFrame)
        assert result["A"].to_list() == [1, 2, 3]
        assert result["D"].to_list() == [1, 2, 3]

        # Test taking the modulo of the DataFrame by a sequence element-wise along the columns (axis='columns')
        result = mixin._df_mod(df_0[["A", "D"]], [1, 2], axis="columns")
        assert isinstance(result, pl.DataFrame)
        assert result["A"].to_list() == [0, 0, 0]
        assert result["D"].to_list() == [1, 0, 1]

        # Test taking the modulo of DataFrames with index-column alignment
        df_1 = df_1.with_columns(D=pl.col("E"))
        result = mixin._df_mod(
            df_0[["unique_id", "A", "D"]],
            df_1[["unique_id", "A", "D"]],
            axis="index",
            index_cols="unique_id",
        )
        assert isinstance(result, pl.DataFrame)
        assert result["A"].to_list() == [None, None, 3]
        assert result["D"].to_list() == [None, None, 0]

    def test_df_mul(self, mixin: PolarsMixin, df_0: pl.DataFrame, df_1: pl.DataFrame):
        # Test multiplying the DataFrame by a sequence element-wise along the rows (axis='index')
        result = mixin._df_mul(df_0[["A", "D"]], df_1["A"], axis="index")
        assert isinstance(result, pl.DataFrame)
        assert result["A"].to_list() == [4, 10, 18]
        assert result["D"].to_list() == [4, 10, 18]

        # Test multiplying the DataFrame by a sequence element-wise along the columns (axis='columns')
        result = mixin._df_mul(df_0[["A", "D"]], [1, 2], axis="columns")
        assert isinstance(result, pl.DataFrame)
        assert result["A"].to_list() == [1, 2, 3]
        assert result["D"].to_list() == [2, 4, 6]

        # Test multiplying DataFrames with index-column alignment
        df_1 = df_1.with_columns(D=pl.col("E"))
        result = mixin._df_mul(
            df_0[["unique_id", "A", "D"]],
            df_1[["unique_id", "A", "D"]],
            axis="index",
            index_cols="unique_id",
        )
        assert isinstance(result, pl.DataFrame)
        assert result["A"].to_list() == [None, None, 12]
        assert result["D"].to_list() == [None, None, 3]

    def test_df_norm(self, mixin: PolarsMixin):
        df = pl.DataFrame({"A": [3, 4], "B": [4, 3]})
        # If include_cols = False
        norm = mixin._df_norm(df)
        assert isinstance(norm, pl.Series)
        assert len(norm) == 2
        assert norm[0] == 5
        assert norm[1] == 5

        # If include_cols = True
        norm = mixin._df_norm(df, include_cols=True)
        assert isinstance(norm, pl.DataFrame)
        assert len(norm) == 2
        assert norm.columns == ["A", "B", "norm"]
        assert norm.row(0, named=True)["norm"] == 5
        assert norm.row(1, named=True)["norm"] == 5

    def test_df_or(self, mixin: PolarsMixin, df_0: pl.DataFrame, df_1: pl.DataFrame):
        # Test comparing the DataFrame with a sequence element-wise along the rows (axis='index')
        df_0 = df_0.with_columns(F=pl.Series([True, True, False]))
        df_1 = df_1.with_columns(F=pl.Series([False, False, True]))
        result = mixin._df_or(df_0[["C", "F"]], df_1["F"], axis="index")
        assert isinstance(result, pl.DataFrame)
        assert result["C"].to_list() == [True, False, True]
        assert result["F"].to_list() == [True, True, True]

        # Test comparing the DataFrame with a sequence element-wise along the columns (axis='columns')
        result = mixin._df_or(df_0[["C", "F"]], [True, False], axis="columns")
        assert isinstance(result, pl.DataFrame)
        assert result["C"].to_list() == [True, True, True]
        assert result["F"].to_list() == [True, True, False]

        # Test comparing DataFrames with index-column alignment
        result = mixin._df_or(
            df_0[["unique_id", "C", "F"]],
            df_1[["unique_id", "C", "F"]],
            axis="index",
            index_cols="unique_id",
        )
        assert isinstance(result, pl.DataFrame)
        assert result["C"].to_list() == [True, None, True]
        assert result["F"].to_list() == [True, True, False]

    def test_df_reindex(
        self, mixin: PolarsMixin, df_0: pl.DataFrame, df_1: pl.DataFrame
    ):
        # Test with DataFrame
        reindexed = mixin._df_reindex(df_0, df_1, "unique_id")
        assert isinstance(reindexed, pl.DataFrame)
        assert reindexed["unique_id"].to_list() == ["z", "a", "b"]
        assert reindexed["A"].to_list() == [3, None, None]
        assert reindexed["B"].to_list() == ["c", None, None]
        assert reindexed["C"].to_list() == [True, None, None]
        assert reindexed["D"].to_list() == [3, None, None]

        # Test with list
        reindexed = mixin._df_reindex(df_0, ["z", "a", "b"], "unique_id")
        assert isinstance(reindexed, pl.DataFrame)
        assert reindexed["unique_id"].to_list() == ["z", "a", "b"]
        assert reindexed["A"].to_list() == [3, None, None]
        assert reindexed["B"].to_list() == ["c", None, None]
        assert reindexed["C"].to_list() == [True, None, None]
        assert reindexed["D"].to_list() == [3, None, None]

        # Test reindexing with a different column name
        reindexed = mixin._df_reindex(
            df_0,
            ["z", "a", "b"],
            new_index_cols="new_index",
            original_index_cols="unique_id",
        )
        assert isinstance(reindexed, pl.DataFrame)
        assert reindexed["new_index"].to_list() == ["z", "a", "b"]
        assert reindexed["A"].to_list() == [3, None, None]
        assert reindexed["B"].to_list() == ["c", None, None]
        assert reindexed["C"].to_list() == [True, None, None]
        assert reindexed["D"].to_list() == [3, None, None]

    def test_df_rename_columns(self, mixin: PolarsMixin, df_0: pl.DataFrame):
        renamed = mixin._df_rename_columns(df_0, ["A", "B"], ["X", "Y"])
        assert renamed.columns == ["unique_id", "X", "Y", "C", "D"]

    def test_df_reset_index(self, mixin: PolarsMixin, df_0: pl.DataFrame):
        # with drop = False
        new_df = mixin._df_reset_index(df_0)
        assert mixin._df_all(new_df == df_0).all()

        # with drop = True
        new_df = mixin._df_reset_index(df_0, index_cols="unique_id", drop=True)
        assert new_df.columns == ["A", "B", "C", "D"]
        assert len(new_df) == len(df_0)
        for col in new_df.columns:
            assert (new_df[col] == df_0[col]).all()

    def test_df_remove(self, mixin: PolarsMixin, df_0: pl.DataFrame):
        # Test with list
        removed = mixin._df_remove(df_0, [1, 3], "A")
        assert len(removed) == 1
        assert removed["unique_id"].to_list() == ["y"]

    def test_df_sample(self, mixin: PolarsMixin, df_0: pl.DataFrame):
        # Test with n
        sampled = mixin._df_sample(df_0, n=2, seed=42)
        assert len(sampled) == 2

        # Test with frac
        sampled = mixin._df_sample(df_0, frac=2 / 3, seed=42)
        assert len(sampled) == 2

        # Test with replacement
        sampled = mixin._df_sample(df_0, n=4, with_replacement=True, seed=42)
        assert len(sampled) == 4
        assert sampled.n_unique() < 4

    def test_df_set_index(self, mixin: PolarsMixin, df_0: pl.DataFrame):
        index = pl.int_range(len(df_0), eager=True)
        new_df = mixin._df_set_index(df_0, "index", index)
        assert (new_df["index"] == index).all()

    def test_df_with_columns(self, mixin: PolarsMixin, df_0: pl.DataFrame):
        # Test with list
        new_df = mixin._df_with_columns(
            df_0,
            data=[[4, "d"], [5, "e"], [6, "f"]],
            new_columns=["D", "E"],
        )
        assert list(new_df.columns) == ["unique_id", "A", "B", "C", "D", "E"]
        assert new_df["D"].to_list() == [4, 5, 6]
        assert new_df["E"].to_list() == ["d", "e", "f"]

        # Test with pl.DataFrame
        second_df = pl.DataFrame({"D": [4, 5, 6], "E": ["d", "e", "f"]})
        new_df = mixin._df_with_columns(df_0, second_df)
        assert list(new_df.columns) == ["unique_id", "A", "B", "C", "D", "E"]
        assert new_df["D"].to_list() == [4, 5, 6]
        assert new_df["E"].to_list() == ["d", "e", "f"]

        # Test with dictionary
        new_df = mixin._df_with_columns(
            df_0, data={"D": [4, 5, 6], "E": ["d", "e", "f"]}
        )
        assert list(new_df.columns) == ["unique_id", "A", "B", "C", "D", "E"]
        assert new_df["D"].to_list() == [4, 5, 6]
        assert new_df["E"].to_list() == ["d", "e", "f"]

        # Test with numpy array
        new_df = mixin._df_with_columns(df_0, data=np.array([4, 5, 6]), new_columns="D")
        assert "D" in new_df.columns
        assert new_df["D"].to_list() == [4, 5, 6]

        # Test with pl.Series
        new_df = mixin._df_with_columns(df_0, pl.Series([4, 5, 6]), new_columns="D")
        assert "D" in new_df.columns
        assert new_df["D"].to_list() == [4, 5, 6]

    def test_srs_constructor(self, mixin: PolarsMixin):
        # Test with list
        srs = mixin._srs_constructor([1, 2, 3], name="test", dtype="int64")
        assert srs.name == "test"
        assert srs.dtype == pl.Int64

        # Test with numpy array
        srs = mixin._srs_constructor(np.array([1, 2, 3]), name="test")
        assert srs.name == "test"
        assert len(srs) == 3

    def test_srs_contains(self, mixin: PolarsMixin):
        srs = [1, 2, 3, 4, 5]

        # Test with single value
        result = mixin._srs_contains(srs, 3)
        assert result.to_list() == [True]

        # Test with list
        result = mixin._srs_contains(srs, [1, 3, 6])
        assert result.to_list() == [True, True, False]

        # Test with numpy array
        result = mixin._srs_contains(srs, np.array([1, 3, 6]))
        assert result.to_list() == [True, True, False]

    def test_srs_range(self, mixin: PolarsMixin):
        # Test with default step
        srs = mixin._srs_range("test", 0, 5)
        assert srs.name == "test"
        assert srs.to_list() == [0, 1, 2, 3, 4]

        # Test with custom step
        srs = mixin._srs_range("test", 0, 10, step=2)
        assert srs.to_list() == [0, 2, 4, 6, 8]

    def test_srs_to_df(self, mixin: PolarsMixin):
        srs = pl.Series("test", [1, 2, 3])
        df = mixin._srs_to_df(srs)
        assert isinstance(df, pl.DataFrame)
        assert df["test"].to_list() == [1, 2, 3]
