import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessClassifier

from src.base import Base


class GP(Base):
    def __init__(self) -> None:
        self.model = GaussianProcessClassifier(
            max_iter_predict=100, n_restarts_optimizer=30, warm_start=True
        )
        self.columns = [
            "home_goals_last_5",
            "home_points_last_5",
            "away_goals_last_5",
            "away_points_last_5",
            "B365H",
            "B365D",
            "B365A",
        ]

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(x[self.columns], y)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return self.model.predict(x[self.columns])

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(x[self.columns])

    def should_bet(self, expected_values: np.ndarray) -> bool:
        return expected_values.max() > 0
