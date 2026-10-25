from argparse import Namespace

import pandas as pd

from src import run_rnn


def test_load_games_adds_season_and_concatenates(monkeypatch):
    monkeypatch.setattr(
        run_rnn.pd,
        "read_csv",
        lambda path: pd.DataFrame({"Date": ["2020-01-01"], "value": [1]}),
    )
    result = run_rnn.load_games([2018, 2019])

    assert result["_season"].tolist() == [2018, 2019]
    assert result["value"].tolist() == [1, 1]


def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr(run_rnn, "DEFAULT_FEATURES", ["FG"])
    monkeypatch.setattr("sys.argv", ["run_rnn"])

    args = run_rnn.parse_args()

    assert args.initial_training_years == [2015, 2016]
    assert args.sequence_length == 10
    assert args.features == ["FG"]


def test_load_odds_uses_project_filename_convention(monkeypatch):
    captured_paths = []

    def fake_read_csv(path):
        captured_paths.append(path)
        return pd.DataFrame({"home_odds": [2.0], "away_odds": [2.0]})

    monkeypatch.setattr(run_rnn.pd, "read_csv", fake_read_csv)
    result = run_rnn.load_odds([2019])

    assert captured_paths == [".\\data\\input\\nba odds 2018-19.csv"]
    assert len(result) == 1
