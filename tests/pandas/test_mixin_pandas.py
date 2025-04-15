import pandas as pd
import pytest

from mesa_frames.concrete.pandas.mixin import PandasMixin


@pytest.fixture
def df_or():
    return PandasMixin()._df_or


@pytest.fixture
def df_0():
    return pd.DataFrame(
        {
            "unique_id": ["x", "y", "z"],
            "A": [1, 0, 1],
            "B": ["a", "b", "c"],
            "C": [True, False, True],
            "D": [0, 1, 1],
        }
    ).set_index("unique_id")


@pytest.fixture
def df_1():
    return pd.DataFrame(
        {
            "unique_id": ["z", "a", "b"],
            "A": [0, 1, 0],
            "B": ["d", "e", "f"],
            "C": [False, True, False],
            "E": [1, 0, 1],
        }
    ).set_index("unique_id")


def test_df_or(df_or: df_or, df_0: pd.DataFrame, df_1: pd.DataFrame):
    # Test comparing the DataFrame with a sequence element-wise along the rows (axis='index')
    df_0["F"] = [True, True, False]
    df_1["F"] = [False, False, True]
    result = df_or(df_0[["C", "F"]], df_1["F"], axis="index")
    assert isinstance(result, pd.DataFrame)
    assert result["C"].tolist() == [True, False, True]
    assert result["F"].tolist() == [True, True, True]

    # Test comparing the DataFrame with a sequence element-wise along the columns (axis='columns')
    result = df_or(df_0[["C", "F"]], [True, False], axis="columns")
    assert isinstance(result, pd.DataFrame)
    assert result["C"].tolist() == [True, True, True]
    assert result["F"].tolist() == [True, True, False]

    # Test comparing DataFrames with index-column alignment
    result = df_or(
        df_0[["C", "F"]],
        df_1[["C", "F"]],
        axis="index",
        index_cols="unique_id",
    )
    assert isinstance(result, pd.DataFrame)
    assert result["C"].tolist() == [True, False, True]
    assert result["F"].tolist() == [True, True, False]
