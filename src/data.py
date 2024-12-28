from collections import defaultdict
from typing import Tuple
import pandas as pd


import os

from src.elo import EloOnly

mapping = ["H", "D", "A"]


def get_data(
    num_test_matches: int,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    df = get_raw_data()
    df = df[
        [
            "FTR",
            "FTHG",
            "FTAG",
            "HomeTeam",
            "AwayTeam",
            "Date",
            "days_since_first_game",
            "B365H",
            "B365D",
            "B365A",
        ]
    ].copy()
    df = df[~pd.isna(df["FTR"])]
    assert len(df[df.isna().any(axis=1)]) == 0
    y = df["FTR"].apply(lambda x: mapping.index(x))
    df = compute_current_form_last_n_games(df, 5)
    df = compute_current_form_last_n_games(df, 3)
    df = compute_elo(df)

    x = df.drop(columns=["FTR", "FTHG", "FTAG"])

    # Most recent games
    test_indices = df.index.isin(df.sort_values("Date").tail(num_test_matches).index)

    x_train = x[~test_indices]
    y_train = y[~test_indices]
    x_test = x[test_indices]
    y_test = y[test_indices]
    return x_train, y_train, x_test, y_test


def get_raw_data() -> pd.DataFrame:
    # Raw data csvs
    # can be downloaded from https://www.football-data.co.uk/englandm.php
    # or run the script src/download_data.py
    dfs = []
    for f in os.listdir("."):
        if f.endswith(".csv"):
            sub_df = pd.read_csv(f)
            sub_df["file"] = f
            dfs.append(sub_df)
    df = pd.concat(dfs).reset_index(drop=True)

    df["HomeTeam"] = df["HomeTeam"].astype("category")
    df["AwayTeam"] = df["AwayTeam"].astype("category")
    df["Referee"] = df["Referee"].astype("category")
    def parse_date(date_str):
        try:
            # Try parsing with dd/mm/yy
            return pd.to_datetime(date_str, format='%d/%m/%y')
        except ValueError:
            # Fallback to dd/mm/yyyy
            return pd.to_datetime(date_str, format='%d/%m/%Y')
    df["Date"] = df["Date"].apply(parse_date)
    df.sort_values("Date", inplace=True)
    df["days_since_first_game"] = (df["Date"] - df["Date"].min()).dt.days
    return df


def compute_elo(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    elo_model = EloOnly(k_factor=20, home_advantage=200)
    y = df["FTR"].apply(lambda x: mapping.index(x))
    pregame_elos = elo_model.fit(df, y)
    df["HomeElo"] = pd.NA
    df["AwayElo"] = pd.NA
    home_elo_index = df.columns.get_loc("HomeElo")
    away_elo_index = df.columns.get_loc("AwayElo")
    for i, (home_elo, away_elo) in enumerate(pregame_elos):
        df.iloc[i, home_elo_index] = home_elo
        df.iloc[i, away_elo_index] = away_elo

    df["HomeElo"] = df["HomeElo"].astype(float)
    df["AwayElo"] = df["AwayElo"].astype(float)

    return df


def compute_current_form_last_n_games(df: pd.DataFrame, n: int) -> pd.DataFrame:
    df = df.copy()
    df[f"home_goals_last_{n}"] = 0
    df[f"home_points_last_{n}"] = 0
    df[f"away_goals_last_{n}"] = 0
    df[f"away_points_last_{n}"] = 0

    for idx, row in df.iterrows():
        last_n_games_home_team = (
            df[
                (
                    (df["HomeTeam"] == row["HomeTeam"])
                    | (df["AwayTeam"] == row["HomeTeam"])
                )
                & (df["Date"] < row["Date"])
            ]
            .sort_values("Date")
            .tail(n)
        )
        for _, game in last_n_games_home_team.iterrows():
            if game["HomeTeam"] == row["HomeTeam"]:
                df.at[idx, f"home_goals_last_{n}"] += game["FTHG"]
                df.at[idx, f"home_points_last_{n}"] += (
                    3 if game["FTR"] == "H" else 1 if game["FTR"] == "D" else 0
                )
            else:
                df.at[idx, f"home_goals_last_{n}"] += game["FTAG"]
                df.at[idx, f"home_points_last_{n}"] += (
                    3 if game["FTR"] == "A" else 1 if game["FTR"] == "D" else 0
                )

        last_n_games_away_team = (
            df[
                (
                    (df["HomeTeam"] == row["AwayTeam"])
                    | (df["AwayTeam"] == row["AwayTeam"])
                )
                & (df["Date"] < row["Date"])
            ]
            .sort_values("Date")
            .tail(n)
        )
        for _, game in last_n_games_away_team.iterrows():
            if game["HomeTeam"] == row["AwayTeam"]:
                df.at[idx, f"away_goals_last_{n}"] += game["FTHG"]
                df.at[idx, f"away_points_last_{n}"] += (
                    3 if game["FTR"] == "H" else 1 if game["FTR"] == "D" else 0
                )
            else:
                df.at[idx, f"away_goals_last_{n}"] += game["FTAG"]
                df.at[idx, f"away_points_last_{n}"] += (
                    3 if game["FTR"] == "A" else 1 if game["FTR"] == "D" else 0
                )
    return df
