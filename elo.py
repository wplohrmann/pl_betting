import pandas as pd
import numpy as np

class EloOnly:
    def __init__(self, k_factor, home_advantage):
        self.k_factor = k_factor
        self.initial_elo = 1500
        self.home_advantage = home_advantage
        self.elo_ratings = {}

    def _expected_score(self, elo_a, elo_b):
        """Calculate the expected score for team A against team B."""
        return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

    def _update_elo(self, elo_a, elo_b, actual_score):
        """Update ELO rating for a single team based on the match result."""
        expected_score = self._expected_score(elo_a, elo_b)
        return elo_a + self.k_factor * (actual_score - expected_score)

    def fit(self, x: pd.DataFrame, y: pd.Series):
        """Fit the model to the match results.
        x: DataFrame with columns ["HomeTeam", "AwayTeam"]
        y: Series with values 0 (home win), 1 (draw), 2 (away win)
        """
        pregame_elos = []
        for _, row in x.iterrows():
            home_team, away_team = row["HomeTeam"], row["AwayTeam"]
            result = y.loc[_]

            # Initialize ELO ratings if teams are new
            if home_team not in self.elo_ratings:
                self.elo_ratings[home_team] = self.initial_elo
            if away_team not in self.elo_ratings:
                self.elo_ratings[away_team] = self.initial_elo

            pregame_elos.append((self.elo_ratings[home_team], self.elo_ratings[away_team]))
            home_elo = self.elo_ratings[home_team] + self.home_advantage
            away_elo = self.elo_ratings[away_team]

            # Determine the actual scores for ELO calculation
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

    def predict(self, x: pd.DataFrame):
        """Predict the outcome of matches.
        x: DataFrame with columns ["HomeTeam", "AwayTeam"]
        Returns: Array of predictions (0: home win, 1: draw, 2: away win)
        """
        predictions = []
        for _, row in x.iterrows():
            home_team, away_team = row["HomeTeam"], row["AwayTeam"]
            home_elo = self.elo_ratings.get(home_team, self.initial_elo) + self.home_advantage
            away_elo = self.elo_ratings.get(away_team, self.initial_elo)

            # Calculate win probabilities
            prob_home_win = self._expected_score(home_elo, away_elo)
            prob_away_win = self._expected_score(away_elo, home_elo)
            prob_draw = 1 - prob_home_win - prob_away_win

            # Predict based on highest probability
            probs = [prob_home_win, prob_draw, prob_away_win]
            predictions.append(np.argmax(probs))

        return np.array(predictions)

    def predict_proba(self, x: pd.DataFrame):
        """Predict probabilities for match outcomes.
        x: DataFrame with columns ["HomeTeam", "AwayTeam"]
        Returns: Array of probabilities for each outcome.
        """
        probabilities = []
        for _, row in x.iterrows():
            home_team, away_team = row["HomeTeam"], row["AwayTeam"]
            home_elo = self.elo_ratings.get(home_team, self.initial_elo) + self.home_advantage
            away_elo = self.elo_ratings.get(away_team, self.initial_elo)

            # Calculate win probabilities
            prob_home_win = self._expected_score(home_elo, away_elo)
            prob_away_win = self._expected_score(away_elo, home_elo)
            prob_draw = 1 - prob_home_win - prob_away_win

            probabilities.append([prob_home_win, prob_draw, prob_away_win])

        return np.array(probabilities)

    def should_bet(self, expected_values):
        return expected_values.max() > 0
