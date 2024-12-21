import abc
from typing import Any
import numpy as np
import pandas as pd


class Base:
    @abc.abstractmethod
    def fit(self, x: pd.DataFrame, y: pd.Series) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, x: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def should_bet(self, expected_values: np.ndarray) -> bool:
        raise NotImplementedError
