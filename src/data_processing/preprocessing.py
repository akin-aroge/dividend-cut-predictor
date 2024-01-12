""" List of functions for data processing. """

from typing import Literal, Sequence
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils import resample
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class NARoWRemover(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_check, how: Literal["any", "all"] = None) -> None:
        self.cols_to_check = cols_to_check
        if how is None:
            self.how = "all"
        else:
            self.how = how

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        n_rows = len(X)
        X = X.dropna(subset=self.cols_to_check, how=self.how)
        new_n_rows = len(X)
        n_rows_dropped = n_rows - new_n_rows
        # logger.info(f'dropped {n_rows_dropped} rows')
        logging.getLogger(self.__class__.__name__).info(
            f"dropped {n_rows_dropped} rows with NA in columns: {self.cols_to_check}"
        )
        return X


class ColumnsRemover(BaseEstimator, TransformerMixin):
    def __init__(self, col_names) -> None:
        self.col_names = col_names

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X = X.drop(self.col_names, axis=1)
        return X


class BinarizeCol(BaseEstimator, TransformerMixin):
    def __init__(self, col_name, true_val) -> None:
        self.col_name = col_name
        self.true_val = true_val

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X[self.col_name] = np.where(X[self.col_name] == self.true_val, 1, 0)
        logging.getLogger(self.__class__.__name__).info(f"binarized {self.col_name}")
        return X


class SMOTEBalancer(BaseEstimator, TransformerMixin):
    def __init__(self, label_col_name: str, random_state=None) -> None:
        self.random_state = random_state
        self.label_col_name = label_col_name
        self.X_resampled = None
        self.y_resampled = None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X, y = self._split_Xy(X, label_col_name=self.label_col_name)
        X_resampled, y_resampled = SMOTE(random_state=self.random_state).fit_resample(
            X, y
        )

        initial_label_counts = y.value_counts().to_dict()
        new_label_counts = y_resampled.value_counts().to_dict()
        logger.info(
            f"balanced data, initial label counts:{initial_label_counts} | new label counts:{new_label_counts}"
        )

        resampled_data = X_resampled
        resampled_data[self.label_col_name] = y_resampled

        return resampled_data

    def _split_Xy(df: pd.DataFrame, label_col_name: str):
        X = df.drop(label_col_name, axis=1)
        y = df[label_col_name]

        return X, y
