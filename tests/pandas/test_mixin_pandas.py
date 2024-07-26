import numpy as np
import pandas as pd
import pytest

from mesa_frames.concrete.pandas.mixin import PandasMixin


class TestPandasMixin:
    @pytest.fixture
    def mixin(self):
        return PandasMixin()

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {"A": [1, 2, 3], "B": ["a", "b", "c"], "C": [True, False, True]},
            index=pd.Index(["x", "y", "z"], name="unique_id"),
        )

    def test_df_column_names(self, mixin: PandasMixin, sample_df: pd.DataFrame):
        assert set(mixin._df_column_names(sample_df)) == {"A", "B", "C", "unique_id"}

    def test_df_combine_first(self, mixin: PandasMixin):
        df1 = pd.DataFrame(
            {"A": [1, np.nan, 3], "B": [4, 5, 6]},
            index=pd.Index(["x", "y", "z"], name="unique_id"),
        )
        df2 = pd.DataFrame(
            {"A": [10, 20, 30], "B": [40, 50, 60]},
            index=pd.Index(["x", "y", "z"], name="unique_id"),
        )
        result = mixin._df_combine_first(
            df1,
            df2,
            index_col="unique_id",
        )
        expected = pd.DataFrame(
            {"A": [1, 20, 3], "B": [4, 5, 6]},
            index=pd.Index(["x", "y", "z"], name="unique_id"),
        )
        pd.testing.assert_frame_equal(result, expected, check_dtype=False)

    def test_df_concat(self, mixin: PandasMixin, sample_df: pd.DataFrame):
        df1 = sample_df
        df2 = pd.DataFrame({"A": [4, 5], "B": ["d", "e"], "C": [False, True]})

        ## Test vertical concatenation
        # With ignore_index = False
        vertical = mixin._df_concat([df1, df2], how="vertical")
        assert len(vertical) == 5
        assert vertical.index.tolist() == ["x", "y", "z", 0, 1]

        # With ignore_index = True
        vertical_ignore_index = mixin._df_concat(
            [df1, df2], how="vertical", ignore_index=True
        )
        assert len(vertical_ignore_index) == 5
        assert vertical_ignore_index.index.tolist() == list(range(5))

        ## Test horizontal concatenation
        # With ignore_index = False
        horizontal = mixin._df_concat([df1, df2], how="horizontal")
        assert len(horizontal.columns) == 6
        assert horizontal.columns.to_list() == ["A", "B", "C", "A", "B", "C"]

        # With ignore_index = True
        horizontal = mixin._df_concat([df1, df2], how="horizontal", ignore_index=True)
        assert len(horizontal.columns) == 6
        assert horizontal.columns.to_list() == list(range(6))

    def test_df_constructor(self, mixin: PandasMixin):
        # Test with list of lists
        data = [[1, "a"], [2, "b"], [3, "c"]]
        df = mixin._df_constructor(
            data, columns=["num", "letter"], dtypes={"num": "int64"}
        )
        assert list(df.columns) == ["num", "letter"]
        assert df["num"].dtype == "int64"
        assert df["num"].to_list() == [1, 2, 3]
        assert df["letter"].to_list() == ["a", "b", "c"]

        # Test with dictionary
        data = {"num": [1, 2, 3], "letter": ["a", "b", "c"]}
        df = mixin._df_constructor(data)
        assert list(df.columns) == ["num", "letter"]
        assert df["num"].tolist() == [1, 2, 3]
        assert df["letter"].tolist() == ["a", "b", "c"]

        # Test with index_col
        df = mixin._df_constructor(data, index_col="num")
        assert df.index.name == "num"
        assert df.index.tolist() == [1, 2, 3]

    def test_df_contains(self, mixin: PandasMixin, sample_df: pd.DataFrame):
        # Test with list
        result = mixin._df_contains(sample_df, "A", [1, 3, 5])
        assert result.tolist() == [True, True, False]

    def test_df_filter(self, mixin: PandasMixin, sample_df: pd.DataFrame):
        condition = pd.DataFrame(
            {
                "A": [False, True, True],
                "B": [False, False, True],
                "C": [True, False, True],
            },
            index=pd.Index(["x", "y", "z"], name="unique_id"),
        )

        # Test with pd.DataFrame and all=True
        filtered = mixin._df_filter(sample_df, condition, all=True)
        assert len(filtered) == 1
        assert filtered.index.tolist() == ["z"]

        # Test with pd.DataFrame and all=False
        filtered = mixin._df_filter(sample_df, condition, all=False)
        assert len(filtered) == 3
        assert filtered.index.tolist() == ["x", "y", "z"]

    def test_df_get_bool_mask(self, mixin: PandasMixin, sample_df: pd.DataFrame):
        # Test with pd.Series[bool]
        mask = mixin._df_get_bool_mask(sample_df, "A", pd.Series([True, False, True]))
        assert mask.tolist() == [True, False, True]
        assert (mask.index == sample_df.index).all()

        # Test with DataFrame
        mask_df = pd.DataFrame({"A": [1, 3]})
        mask = mixin._df_get_bool_mask(sample_df, "A", mask_df)
        assert mask.tolist() == [True, False, True]
        assert (mask.index == sample_df.index).all()

        # Test with single value
        mask = mixin._df_get_bool_mask(sample_df, "A", 1)
        assert mask.tolist() == [True, False, False]
        assert (mask.index == sample_df.index).all()

        # Test with list of values
        mask = mixin._df_get_bool_mask(sample_df, "A", [1, 3])
        assert mask.tolist() == [True, False, True]
        assert (mask.index == sample_df.index).all()

        # Test with negate=True
        mask = mixin._df_get_bool_mask(sample_df, "A", [1, 3], negate=True)
        assert mask.tolist() == [False, True, False]
        assert (mask.index == sample_df.index).all()

    def test_df_get_masked_df(self, mixin: PandasMixin, sample_df: pd.DataFrame):
        # Test with pd.Series[bool]
        masked_df = mixin._df_get_masked_df(
            sample_df, "A", pd.Series([True, False, True])
        )
        assert masked_df["A"].tolist() == [1, 3]
        assert masked_df.index.tolist() == ["x", "z"]

        # Test with DataFrame
        mask_df = pd.DataFrame({"A": [1, 3]})
        masked_df = mixin._df_get_masked_df(sample_df, "A", mask_df)
        assert masked_df["A"].tolist() == [1, 3]
        assert masked_df.index.tolist() == ["x", "z"]

        # Test with single value
        masked_df = mixin._df_get_masked_df(sample_df, "A", 1)
        assert masked_df["A"].tolist() == [1]
        assert masked_df.index.tolist() == ["x"]

        # Test with list of values
        masked_df = mixin._df_get_masked_df(sample_df, "A", [1, 3])
        assert masked_df["A"].tolist() == [1, 3]
        assert masked_df.index.tolist() == ["x", "z"]

        # Test with columns
        masked_df = mixin._df_get_masked_df(sample_df, "A", [1, 3], columns=["B"])
        assert list(masked_df.columns) == ["B"]
        assert masked_df["B"].tolist() == ["a", "c"]
        assert masked_df.index.tolist() == ["x", "z"]

        # Test with negate=True
        masked = mixin._df_get_masked_df(sample_df, "A", [1, 3], negate=True)
        assert len(masked) == 1

    def test_df_iterator(self, mixin: PandasMixin, sample_df: pd.DataFrame):
        iterator = mixin._df_iterator(sample_df)
        first_item = next(iterator)
        assert first_item == {"A": 1, "B": "a", "C": True, "unique_id": "x"}

    def test_df_join(self, mixin: PandasMixin):
        left = pd.DataFrame({"A": [1, 2], "B": ["a", "b"]})
        right = pd.DataFrame({"A": [1, 3], "C": ["x", "y"]})

        # Test with 'on' (left join)
        joined = mixin._df_join(left, right, on="A")
        assert list(joined.columns) == ["A", "B", "C"]
        assert joined["A"].tolist() == [1, 2]

        # Test with 'left_on' and 'right_on' (left join)
        right_1 = pd.DataFrame({"D": [1, 2], "C": ["x", "y"]})
        joined = mixin._df_join(left, right_1, left_on="A", right_on="D")
        assert list(joined.columns) == ["A", "B", "D", "C"]
        assert joined["A"].tolist() == [1, 2]

        # Test with 'right' join
        joined = mixin._df_join(left, right, on="A", how="right")
        assert list(joined.columns) == ["A", "B", "C"]
        assert joined["A"].tolist() == [1, 3]

        # Test with 'inner' join
        joined = mixin._df_join(left, right, on="A", how="inner")
        assert list(joined.columns) == ["A", "B", "C"]
        assert joined["A"].tolist() == [1]

        # Test with 'outer' join
        joined = mixin._df_join(left, right, on="A", how="outer")
        assert list(joined.columns) == ["A", "B", "C"]
        assert joined["A"].tolist() == [1, 2, 3]

        # Test with 'cross' join
        joined = mixin._df_join(left, right, how="cross")
        assert list(joined.columns) == ["A", "B", "A_right", "C"]
        assert len(joined) == 4
        assert joined.iloc[0].tolist() == [1, "a", 1, "x"]
        assert joined.iloc[1].tolist() == [1, "a", 3, "y"]
        assert joined.iloc[2].tolist() == [2, "b", 1, "x"]
        assert joined.iloc[3].tolist() == [2, "b", 3, "y"]

        # Test with different 'suffix'
        joined = mixin._df_join(left, right, suffix="_r", how="cross")
        assert list(joined.columns) == ["A", "B", "A_r", "C"]
        assert len(joined) == 4
        assert joined.iloc[0].tolist() == [1, "a", 1, "x"]
        assert joined.iloc[1].tolist() == [1, "a", 3, "y"]
        assert joined.iloc[2].tolist() == [2, "b", 1, "x"]
        assert joined.iloc[3].tolist() == [2, "b", 3, "y"]

    def test_df_norm(self, mixin: PandasMixin):
        df = pd.DataFrame({"A": [3, 4], "B": [4, 3]})
        norm = mixin._df_norm(df)
        assert len(norm) == 2
        assert norm[0] == 5
        assert norm[1] == 5

    def test_df_rename_columns(self, mixin: PandasMixin, sample_df: pd.DataFrame):
        renamed = mixin._df_rename_columns(sample_df, ["A", "B"], ["X", "Y"])
        assert list(renamed.columns) == ["X", "Y", "C"]

    def test_df_remove(self, mixin: PandasMixin, sample_df: pd.DataFrame):
        # Test with list
        removed = mixin._df_remove(sample_df, [1, 3], "A")
        assert len(removed) == 1
        assert removed.index.tolist() == ["y"]

    def test_df_sample(self, mixin: PandasMixin, sample_df: pd.DataFrame):
        # Test with n
        sampled = mixin._df_sample(sample_df, n=2, seed=42)
        assert len(sampled) == 2

        # Test with frac
        sampled = mixin._df_sample(sample_df, frac=0.66, seed=42)
        assert len(sampled) == 2

        # Test with replacement
        sampled = mixin._df_sample(sample_df, n=4, with_replacement=True, seed=42)
        assert len(sampled) == 4

    def test_df_with_columns(self, mixin: PandasMixin, sample_df: pd.DataFrame):
        # Test with list
        new_df = mixin._df_with_columns(
            sample_df, [[4, "d"], [5, "e"], [6, "f"]], ["D", "E"]
        )
        assert list(new_df.columns) == ["A", "B", "C", "D", "E"]
        assert new_df["D"].tolist() == [4, 5, 6]
        assert new_df["E"].tolist() == ["d", "e", "f"]

        # Test with pd.DataFrame
        second_df = pd.DataFrame({"D": [4, 5, 6], "E": ["d", "e", "f"]})
        new_df = mixin._df_with_columns(sample_df, second_df)
        assert list(new_df.columns) == ["A", "B", "C", "D", "E"]
        assert new_df["D"].tolist() == [4, 5, 6]
        assert new_df["E"].tolist() == ["d", "e", "f"]

        # Test with dictionary
        new_df = mixin._df_with_columns(
            sample_df, {"D": [4, 5, 6], "E": ["d", "e", "f"]}
        )
        assert list(new_df.columns) == ["A", "B", "C", "D", "E"]
        assert new_df["D"].tolist() == [4, 5, 6]
        assert new_df["E"].tolist() == ["d", "e", "f"]

        # Test with numpy array
        new_df = mixin._df_with_columns(sample_df, np.array([4, 5, 6]), "D")
        assert "D" in new_df.columns
        assert new_df["D"].tolist() == [4, 5, 6]

        # Test with pandas Series
        new_df = mixin._df_with_columns(sample_df, pd.Series([4, 5, 6]), "D")
        assert "D" in new_df.columns
        assert new_df["D"].tolist() == [4, 5, 6]

    def test_srs_constructor(self, mixin: PandasMixin):
        # Test with list
        srs = mixin._srs_constructor([1, 2, 3], name="test", dtype="int64")
        assert srs.name == "test"
        assert srs.dtype == "int64"

        # Test with numpy array
        srs = mixin._srs_constructor(np.array([1, 2, 3]), name="test")
        assert srs.name == "test"
        assert len(srs) == 3

        # Test with custom index
        srs = mixin._srs_constructor([1, 2, 3], name="test", index=["a", "b", "c"])
        assert srs.index.tolist() == ["a", "b", "c"]

    def test_srs_contains(self, mixin: PandasMixin):
        srs = pd.Series([1, 2, 3, 4, 5])

        # Test with single value
        result = mixin._srs_contains(srs, 3)
        assert result.tolist() == [True]

        # Test with list
        result = mixin._srs_contains(srs, [1, 3, 6])
        assert result.tolist() == [True, True, False]

        # Test with numpy array
        result = mixin._srs_contains(srs, np.array([1, 3, 6]))
        assert result.tolist() == [True, True, False]

    def test_srs_range(self, mixin: PandasMixin):
        # Test with default step
        srs = mixin._srs_range("test", 0, 5)
        assert srs.name == "test"
        assert srs.tolist() == [0, 1, 2, 3, 4]

        # Test with custom step
        srs = mixin._srs_range("test", 0, 10, step=2)
        assert srs.tolist() == [0, 2, 4, 6, 8]
