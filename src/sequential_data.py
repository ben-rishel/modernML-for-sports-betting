"""Utilities for turning chronological NBA games into RNN-ready sequences."""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SequentialDataset:
    """Sequences, labels, and game metadata produced for an RNN."""

    X: np.ndarray
    y: np.ndarray
    metadata: pd.DataFrame


class SequentialDataBuilder:
    """Build team-history sequences without using the current game's statistics.

    The input dataframe must be ordered chronologically or contain a ``Date``
    column.  For each feature name supplied, the dataframe must contain
    ``home_<feature>`` and ``away_<feature>`` columns.  Each timestep contains
    the selected statistics for the home team followed by those for the away
    team.  A game is included only after both teams have ``sequence_length``
    previous games.
    """

    def __init__(
        self,
        feature_columns: List[str],
        sequence_length: int = 10,
        target_column: str = "home_victory",
        date_column: str = "Date",
        home_team_column: str = "Home/Neutral",
        away_team_column: str = "Visitor/Neutral",
    ):
        if not feature_columns:
            raise ValueError("feature_columns must contain at least one feature.")
        if sequence_length < 1:
            raise ValueError("sequence_length must be at least 1.")

        self.feature_columns = list(feature_columns)
        self.sequence_length = sequence_length
        self.target_column = target_column
        self.date_column = date_column
        self.home_team_column = home_team_column
        self.away_team_column = away_team_column

    def build(self, games: pd.DataFrame) -> SequentialDataset:
        """Create RNN inputs from a dataframe of games.

        Returns:
            SequentialDataset: ``X`` has shape
                ``(samples, sequence_length, 2 * num_features)`` and ``y`` has
                shape ``(samples,)``.

        Raises:
            ValueError: If required columns are missing or a selected feature
                contains non-numeric values.
        """
        self._validate_columns(games)
        ordered_games = self._sort_games(games)
        target = self._get_target(ordered_games)

        histories: dict[str, List[np.ndarray]] = {}
        sequences: List[np.ndarray] = []
        labels: List[int] = []
        metadata_rows: List[dict] = []

        for row_position, (_, game) in enumerate(ordered_games.iterrows()):
            home_team = game[self.home_team_column]
            away_team = game[self.away_team_column]
            if pd.isna(home_team) or pd.isna(away_team):
                raise ValueError("Team columns must not contain missing values.")

            home_history = histories.setdefault(str(home_team), [])
            away_history = histories.setdefault(str(away_team), [])
            if (
                len(home_history) >= self.sequence_length
                and len(away_history) >= self.sequence_length
            ):
                home_sequence = home_history[-self.sequence_length :]
                away_sequence = away_history[-self.sequence_length :]
                sequences.append(
                    np.concatenate(
                        [
                            np.stack(home_sequence, axis=0),
                            np.stack(away_sequence, axis=0),
                        ],
                        axis=1,
                    )
                )
                labels.append(int(target.iloc[row_position]))
                metadata_rows.append(
                    {
                        "source_index": game.name,
                        self.date_column: game[self.date_column]
                        if self.date_column in ordered_games
                        else row_position,
                        self.home_team_column: home_team,
                        self.away_team_column: away_team,
                    }
                )

            home_history.append(self._team_features(game, "home"))
            away_history.append(self._team_features(game, "away"))

        num_features = 2 * len(self.feature_columns)
        X = np.asarray(
            sequences, dtype=np.float32
        ).reshape((-1, self.sequence_length, num_features))
        y = np.asarray(labels, dtype=np.int64)
        metadata = pd.DataFrame(metadata_rows)
        return SequentialDataset(X=X, y=y, metadata=metadata)

    def _validate_columns(self, games: pd.DataFrame) -> None:
        required = {
            self.home_team_column,
            self.away_team_column,
            *(
                column
                for feature in self.feature_columns
                for column in (f"home_{feature}", f"away_{feature}")
            ),
        }
        missing = sorted(required.difference(games.columns))
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if games.empty:
            raise ValueError("games must contain at least one row.")

    def _sort_games(self, games: pd.DataFrame) -> pd.DataFrame:
        if self.date_column not in games.columns:
            return games.copy()

        ordered = games.copy()
        parsed_dates = pd.to_datetime(ordered[self.date_column], errors="coerce")
        if parsed_dates.isna().any():
            raise ValueError(f"{self.date_column} contains invalid dates.")
        ordered["_sequential_date"] = parsed_dates
        ordered = ordered.sort_values("_sequential_date", kind="stable")
        return ordered.drop(columns="_sequential_date")

    def _get_target(self, games: pd.DataFrame) -> pd.Series:
        if self.target_column in games.columns:
            target = games[self.target_column]
        elif {"PTS", "PTS.1"}.issubset(games.columns):
            target = (games["PTS.1"] > games["PTS"]).astype(int)
        else:
            raise ValueError(
                f"Missing target column '{self.target_column}'. "
                "Provide it or include PTS and PTS.1."
            )

        if target.isna().any():
            raise ValueError("The target column must not contain missing values.")
        if not target.isin([0, 1]).all():
            raise ValueError("The target column must contain only 0 and 1.")
        return target.reset_index(drop=True)

    def _team_features(self, game: pd.Series, side: str) -> np.ndarray:
        values = game[[f"{side}_{feature}" for feature in self.feature_columns]]
        numeric_values = pd.to_numeric(values, errors="coerce")
        if numeric_values.isna().any():
            raise ValueError(
                f"Selected features for the {side} team must be numeric and non-missing."
            )
        return numeric_values.to_numpy(dtype=np.float32)


def build_sequences(
    games: pd.DataFrame,
    feature_columns: List[str],
    sequence_length: int = 10,
    target_column: str = "home_victory",
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Convenience wrapper returning ``X``, ``y``, and game metadata."""

    dataset = SequentialDataBuilder(
        feature_columns=feature_columns,
        sequence_length=sequence_length,
        target_column=target_column,
    ).build(games)
    return dataset.X, dataset.y, dataset.metadata


def build_row_sequences(
    features: pd.DataFrame,
    labels: pd.Series | np.ndarray | List[int],
    sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Create chronological windows from parent-pipeline feature rows.

    The row immediately preceding a label is the final timestep in its
    sequence, so the current game's features are never included in its input.
    This adapter lets the RNN consume the same engineered and scaled rows as
    the tabular models.
    """

    if sequence_length < 1:
        raise ValueError("sequence_length must be at least 1.")
    if features.empty:
        raise ValueError("features must contain at least one row.")
    values = features.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("features must contain only finite numeric values.")
    target = np.asarray(labels, dtype=np.int64)
    if target.ndim != 1 or target.shape[0] != values.shape[0]:
        raise ValueError("features and labels must contain the same number of rows.")
    if not np.isin(target, [0, 1]).all():
        raise ValueError("labels must contain only 0 and 1.")
    if values.shape[0] <= sequence_length:
        raise ValueError("At least sequence_length + 1 rows are required.")

    X = np.stack(
        [values[start : start + sequence_length] for start in range(values.shape[0] - sequence_length)],
        axis=0,
    )
    y = target[sequence_length:]
    metadata = pd.DataFrame(
        {
            "source_position": np.arange(sequence_length, values.shape[0]),
            "source_index": features.index.to_numpy()[sequence_length:],
        }
    )
    return X, y, metadata
