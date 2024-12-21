from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import log_loss

from src.base import Base

class EloOnly(Base):
    def __init__(self, k_factor: float, home_advantage: float, max_draw_prob: float = 0.3, draw_width: float = 400) -> None:
        self.k_factor = k_factor
        self.initial_elo = 1500
        self.home_advantage = home_advantage
        self.max_draw_prob = max_draw_prob
        self.draw_width = draw_width
        self.elo_ratings: Dict[str, float] = {}

    def _expected_scores(self, elo_a: float, elo_b: float) -> Tuple[float, float, float]:
        """Calculate probabilities for home win, draw, and away win."""
        prob_home_win = 1 / (1 + 10 ** ((elo_b - elo_a) / self.draw_width))
        prob_away_win = 1 - prob_home_win
        prob_draw = self._draw_probability(elo_a, elo_b)

        # Adjust home and away probabilities to account for draw
        prob_home_win *= (1 - prob_draw)
        prob_away_win *= (1 - prob_draw)

        return prob_home_win, prob_draw, prob_away_win

    def _draw_probability(self, elo_a: float, elo_b: float)-> float:
        """Calculate the draw probability based on the Elo difference."""
        elo_diff = abs(elo_a - elo_b)
        return self.max_draw_prob * np.exp(-elo_diff / 400)  # Decrease with larger Elo difference

    def _update_elo(self, elo_a: float, elo_b:float, actual_score: float) -> float:
        """Update ELO rating for a single team based on the match result."""
        expected_score = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
        return elo_a + self.k_factor * (actual_score - expected_score)

    def _fit_fixed_hyperparameters(self, x: pd.DataFrame, y: pd.Series) -> List[Tuple[float, float]]:
        """Fit the model to the match results.
        x: DataFrame with columns ["HomeTeam", "AwayTeam"]
        y: Series with values 0 (home win), 1 (draw), 2 (away win)
        """
        self.elo_ratings = {}
        pregame_elos: List[Tuple[float, float]] = []
        for _, row in x.iterrows():
            home_team, away_team = row["HomeTeam"], row["AwayTeam"]
            result = y.loc[_]

            # Initialize ELO ratings if teams are new
            if home_team not in self.elo_ratings:
                self.elo_ratings[home_team] = min(self.elo_ratings.values(), default=self.initial_elo)
            if away_team not in self.elo_ratings:
                self.elo_ratings[away_team] = min(self.elo_ratings.values(), default=self.initial_elo)

            pregame_elos.append(
                (self.elo_ratings[home_team], self.elo_ratings[away_team])
            )
            home_elo = self.elo_ratings[home_team] + self.home_advantage
            away_elo = self.elo_ratings[away_team]

            # Determine the actual scores for ELO calculation
            home_score: float
            away_score: float
            if result == 0:  # Home win
                home_score, away_score = 1, 0
            elif result == 1:  # Draw
                home_score, away_score = 0.5, 0.5
            else:  # Away win
                home_score, away_score = 0, 1

            # Update ELO ratings
            new_home_elo = self._update_elo(home_elo, away_elo, home_score)
            new_away_elo = self._update_elo(away_elo, home_elo, away_score)

            self.elo_ratings[home_team] = new_home_elo - self.home_advantage
            self.elo_ratings[away_team] = new_away_elo

        return pregame_elos

    def fit(self, x: pd.DataFrame, y: pd.Series) -> List[Tuple[float, float]]:
        def objective(params: List[float]) -> float:
            self.k_factor, self.home_advantage, self.max_draw_prob, self.draw_width = params
            self._fit_fixed_hyperparameters(x, y)
            y_pred_proba = self.predict_proba(x)
            return log_loss(y, y_pred_proba)

        initial_params = [20, 200, 0.3, 400]
        bounds = [(1, 100), (0, 500), (0.01, 0.5), (20, 700)]

        # result = minimize(objective, initial_params, bounds=bounds, method='L-BFGS-B')
        # self.k_factor, self.home_advantage, self.max_draw_prob, self.draw_width = result.x
        self.k_factor, self.home_advantage, self.max_draw_prob, self.draw_width = 7.87167833,  35.02435776,   0.28850455, 242.1318268

        return self._fit_fixed_hyperparameters(x, y)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Predict the outcome of matches.
        x: DataFrame with columns ["HomeTeam", "AwayTeam"]
        Returns: Array of predictions (0: home win, 1: draw, 2: away win)
        """
        predictions = []
        for _, row in x.iterrows():
            home_team, away_team = row["HomeTeam"], row["AwayTeam"]
            home_elo = (
                self.elo_ratings.get(home_team, self.initial_elo) + self.home_advantage
            )
            away_elo = self.elo_ratings.get(away_team, self.initial_elo)

            # Get probabilities
            prob_home_win, prob_draw, prob_away_win = self._expected_scores(home_elo, away_elo)
            probs = [prob_home_win, prob_draw, prob_away_win]
            predictions.append(np.argmax(probs))

        return np.array(predictions)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Predict probabilities for match outcomes.
        x: DataFrame with columns ["HomeTeam", "AwayTeam"]
        Returns: Array of probabilities for each outcome.
        """
        probabilities = np.zeros((len(x), 3))
        for i, (_, row) in enumerate(x.iterrows()):
            home_team, away_team = row["HomeTeam"], row["AwayTeam"]
            home_elo = (
                self.elo_ratings.get(home_team, self.initial_elo) + self.home_advantage
            )
            away_elo = self.elo_ratings.get(away_team, self.initial_elo)

            # Get probabilities
            prob_home_win, prob_draw, prob_away_win = self._expected_scores(home_elo, away_elo)
            probabilities[i] = [prob_home_win, prob_draw, prob_away_win]

        return probabilities

    def should_bet(self, expected_values: np.ndarray) -> bool:
        return expected_values.max() > 0
