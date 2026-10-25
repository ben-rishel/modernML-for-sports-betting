import numpy as np
import pandas as pd
import pytest

from src.sequential_data import SequentialDataBuilder, build_sequences


def _games():
    return pd.DataFrame(
        {
            "Date": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"],
            "Home/Neutral": ["A", "C", "B", "A"],
            "Visitor/Neutral": ["B", "A", "C", "B"],
            "home_stat": [10, 20, 30, 40],
            "away_stat": [1, 2, 3, 4],
            "home_victory": [1, 0, 1, 0],
        }
    )


def test_build_sequences_uses_only_previous_games():
    dataset = SequentialDataBuilder(["stat"], sequence_length=1).build(_games())

    assert dataset.X.shape == (2, 1, 2)
    np.testing.assert_array_equal(dataset.y, [1, 0])
    np.testing.assert_array_equal(dataset.X[:, 0, :], [[1, 20], [2, 30]])
    assert dataset.metadata["Date"].tolist() == ["2020-01-03", "2020-01-04"]


def test_wrapper_returns_same_dataset_parts():
    expected = SequentialDataBuilder(["stat"], sequence_length=1).build(_games())
    actual = build_sequences(_games(), ["stat"], sequence_length=1)

    np.testing.assert_array_equal(actual[0], expected.X)
    np.testing.assert_array_equal(actual[1], expected.y)
    pd.testing.assert_frame_equal(actual[2], expected.metadata)


def test_builder_rejects_missing_columns():
    with pytest.raises(ValueError, match="Missing required columns"):
        SequentialDataBuilder(["missing"], sequence_length=1).build(_games())


def test_builder_rejects_invalid_target():
    games = _games()
    games.loc[0, "home_victory"] = 2

    with pytest.raises(ValueError, match="only 0 and 1"):
        SequentialDataBuilder(["stat"], sequence_length=1).build(games)
