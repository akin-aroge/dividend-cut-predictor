""" List of functions for data transformation. """

from typing import Literal, Sequence
from numpy import ndarray

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils import resample
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from src.modelling import training as train
import pandas as pd
import numpy as np
import logging
from src.utils import utils

logger = logging.getLogger(__name__)



class CollinearColsRemover(BaseEstimator, TransformerMixin):

    def __init__(self, thresh, label_col) -> None:
        self.thresh = thresh
        self.label_col = label_col
    
    
    def fit(self, X:pd.DataFrame, y=None):
        X = X.copy()
        # X.drop(labels=self.label_col, axis=1, inplace=True)
        self.cols_to_drop = self._get_collinear_cols(df=X, thresh=self.thresh)
        return self
    
    def transform(self, X:pd.DataFrame):
        n_cols = X.shape[1]
        X = X.drop(self.cols_to_drop, axis=1, errors='ignore')
        new_n_cols = X.shape[1]
        n_cols_dropped = n_cols - new_n_cols
        # print(type(X))
        logging.getLogger(self.__class__.__name__).info(f'dropped {n_cols_dropped} cols')
        return X

    @staticmethod
    def _get_collinear_cols(df:pd.DataFrame, thresh:np.float_):

        df = df.select_dtypes(include=np.float_)

        corr_mat = df.corr().abs()
        corr_mat_u = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1)
                                    .astype(np.bool_))

        cols_to_drop = [col for col in corr_mat_u.columns \
                    if any(corr_mat_u[col] > thresh)]
        
        return cols_to_drop

class ColumnsOrdinalEncoder(BaseEstimator, TransformerMixin):


    def __init__(self, col_names) -> None:
        self.col_names = col_names
        self.ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        
        # super().__init__(dtype=int)

    def fit(self, X, y=None):
        data_subset = X[self.col_names]
        self.ordinal_encoder.fit(data_subset)
        

        return self
    
    def transform(self, X):
        data_subset = X[self.col_names].copy()
        transformed_data = self.ordinal_encoder.transform(data_subset)

        data = X.copy()
        data[self.col_names] = transformed_data
        logging.getLogger(self.__class__.__name__).info(f'cat.cols. transformed')
        # logging.getLogger(self.__class__.__name__).info(f'cat.cols. encoded: \n {X.head(2)}')

        return data
    
    def inverse_transform(self, X):
        data_subset = X[self.col_names].copy()
        transformed_data = self.ordinal_encoder.inverse_transform(data_subset)
        data = X.copy()
        data[self.col_names] = transformed_data
        return data
    
class  OptimalColumnSelector(BaseEstimator, TransformerMixin):

    def __init__(self, n_min_cols=2, optimal_cols_path=None) -> None:
        super().__init__()
        self.n_min_cols = n_min_cols
        self.optimal_cols_path = optimal_cols_path
        self.optimal_cols = None

    def fit(self, X:pd.DataFrame, y=None):
        if self.optimal_cols_path is not None:
            try:
                optimal_col_names = utils.load_value(self.optimal_cols_path)
            except(FileNotFoundError):
                raise FileNotFoundError('File for optimal columns not available \
                                        optimal columns should be determined first.')
            self.optimal_cols = optimal_col_names
        return self
    
    def transform(self, X:pd.DataFrame):

        X = X.loc[:, self.optimal_cols].copy()

        logging.getLogger(self.__class__.__name__).info(f'selected columns: {self.optimal_cols} ')
        return X

def balance_data(df:pd.DataFrame, label_col_name:str, random_state=None):

    X, y = train.split_Xy(df, label_col_name=label_col_name)

    smote = SMOTE(random_state=random_state)
    X_resample, y_resample = smote.fit_resample(X, y)

    initial_label_counts = y.value_counts().to_dict()
    new_label_counts = y_resample.value_counts().to_dict()
    # logger.info(f"balanced data, initial label counts:{initial_label_counts} | new label counts:{new_label_counts}")

    resampled_data = X_resample
    resampled_data[label_col_name] = y_resample

    logger.info(f"balanced data, initial label counts:{initial_label_counts} \
                 and shape: {df.shape} | new label counts:{new_label_counts} \
                    and shape {resampled_data.shape}")

    return resampled_data