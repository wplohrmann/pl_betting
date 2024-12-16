import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from data import get_data, mapping
from elo import EloOnly


class Simple:
    def fit(self, x, y):
        avg_home = y[y == 0].count() / y.count()
        avg_draw = y[y == 1].count() / y.count()
        avg_away = y[y == 2].count() / y.count()
        assert avg_home + avg_draw + avg_away == 1
        self.probs = [avg_home, avg_draw, avg_away]

    def predict(self, x):
        return np.array([self.probs.index(max(self.probs))] * len(x))

    def predict_proba(self, x):
        return np.array([self.probs] * len(x))

    def should_bet(self, expected_values):
        return True


class Bet365:
    def fit(self, x, y):
        pass

    def predict(self, x):
        odds = x[["B365H", "B365D", "B365A"]].values
        return odds.argmin(axis=1)

    def predict_proba(self, x):
        odds = x[["B365H", "B365D", "B365A"]].values
        raw_probs = 1 / odds
        expected_values = raw_probs * (odds - 1) - (1 - raw_probs)
        assert np.isclose(expected_values, 0).all()
        return raw_probs / np.sum(raw_probs, axis=1, keepdims=True)

    def should_bet(self, expected_values):
        return True

class XGB:
    def __init__(self, threshold, odds_only=False, with_elo=False, **kwargs):
        self.model = XGBClassifier(**kwargs)
        if odds_only:
            self.x_columns = [
                "B365H",
                "B365D",
                "B365A",
            ]
        else:
            self.x_columns = [
                "home_goals_last_5",
                "home_points_last_5",
                "away_goals_last_5",
                "away_points_last_5",

                # "home_goals_last_3",
                # "home_points_last_3",
                # "away_goals_last_3",
                # "away_points_last_3",

                "B365H",
                "B365D",
                "B365A",
            ]
        if with_elo:
            self.x_columns.extend(["AwayElo", "HomeElo"])
        self.threshold = threshold

    def fit(self, x, y):
        self.model.fit(x[self.x_columns], y)

    def predict(self, x):
        return self.model.predict(x[self.x_columns])

    def predict_proba(self, x):
        return self.model.predict_proba(x[self.x_columns])

    def should_bet(self, expected_values):
        return expected_values.max() > self.threshold


thresholds = np.linspace(0, 0.8, 9)
get_xgb_name = lambda t: f"XGBoost ({t:.2f})"
models = {
    **{
        get_xgb_name(threshold): XGB(
            threshold=threshold, with_elo=True, enable_categorical=True, max_depth=3
        )
        for threshold in thresholds
    },
    "Elo (k=20, home advantage=200)": EloOnly(k_factor=20, home_advantage=200),
    "XGB + (no Elo)": XGB(0.0, with_elo=True, enable_categorical=True, max_depth=3),
    "Baseline": Simple(),
    "XGB (odds only)": XGB(0.0, odds_only=True),
    "Bet365": Bet365(),
}

x_train, y_train, x_test, y_test = get_data()
print("Number of test games:", len(x_test))
print("Number of training games:", len(x_train))

metrics = defaultdict(dict)

for name, model in models.items():
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy = (predictions == y_test).sum() / len(y_test)
    earnings = 0
    num_bets = 0
    probs = model.predict_proba(x_test)
    predicted_earnings = 0
    win_probabilities = []
    earnings_if_win = []
    for i, (_, row) in enumerate(x_test.iterrows()):
        probs_this_game = probs[i]
        odds = np.array(
            [row[f"B365{mapping[outcome_to_bet_on]}"] for outcome_to_bet_on in range(3)]
        )
        evs = probs_this_game * (odds - 1) - (1 - probs_this_game)
        if model.should_bet(evs):
            num_bets += 1
            outcome_to_bet_on = np.argmax(evs)
            actual_outcome = y_test.iloc[i]
            if outcome_to_bet_on == actual_outcome:
                earnings += row[f"B365{mapping[outcome_to_bet_on]}"] - 1
            else:
                earnings -= 1
            predicted_earnings += evs.max()
            win_probabilities.append(probs_this_game[outcome_to_bet_on])
            earnings_if_win.append(odds[outcome_to_bet_on] - 1)
    num_samples = 1000
    possible_earnings = np.zeros(num_samples)
    for i in range(num_samples):
        hypothetical_earnings = 0
        samples = np.random.uniform(low=0, high=1, size=len(win_probabilities))
        for sample, win_probability, earning_if_win in zip(
            samples, win_probabilities, earnings_if_win
        ):
            if sample < win_probability:
                hypothetical_earnings += earning_if_win
            else:
                hypothetical_earnings -= 1
        possible_earnings[i] = hypothetical_earnings

    worst_quartile_earnings = np.quantile(possible_earnings, 0.25)
    stddev_earnings = np.std(possible_earnings)

    # TODO: Add the Brier score
    metrics[name]["Accuracy"] = accuracy * 100
    # metrics[name]["Earnings"] = earnings
    metrics[name]["Predicted ROI (%)"] = predicted_earnings / num_bets * 100
    metrics[name]["Standard deviation ROI"] = stddev_earnings / num_bets * 100
    metrics[name]["Worst quartile ROI (%)"] = worst_quartile_earnings / num_bets * 100
    metrics[name]["Number of bets"] = num_bets
    metrics[name]["ROI (%)"] = earnings / num_bets * 100

df = pd.DataFrame(metrics)
print()
print(df.T)

x = thresholds
for column in ["ROI (%)", "Worst quartile ROI (%)"]:
    y = np.array([metrics[get_xgb_name(threshold)][column] for threshold in thresholds])
    plt.scatter(x, y, label=column)

xlim = plt.xlim()
plt.hlines([0], *xlim)
plt.xlim(xlim)
plt.ylabel("ROI (%)")
plt.xlabel("Bet threshold")
plt.legend()
plt.show()
