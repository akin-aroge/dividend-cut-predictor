""" List of functions for data processing. """

from typing import Literal, Sequence
from numpy import ndarray
from scipy.sparse import spmatrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils import resample
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class NARoWRemover(BaseEstimator, TransformerMixin):

    def __init__(self, cols_to_check, how:Literal['any', 'all'] =None) -> None:

        self.cols_to_check = cols_to_check
        if how is None:
            self.how='all'
        else:
            self.how=how

    def fit(self, X, y=None):
        return self
    
    def transform(self, X:pd.DataFrame):
        n_rows = len(X)
        X = X.dropna(subset=self.cols_to_check, how=self.how)
        new_n_rows = len(X)
        n_rows_dropped = n_rows - new_n_rows
        # logger.info(f'dropped {n_rows_dropped} rows')
        logging.getLogger(self.__class__.__name__).info(f'dropped {n_rows_dropped} rows with NA in columns: {self.cols_to_check}')
        return X



class ColumnsRemover(TransformerMixin):

    def __init__(self, col_names) -> None:
        self.col_names = col_names

    def fit(self, X, y=None):
        return self
    
    def transform(self, X:pd.DataFrame):

        X = X.drop(self.col_names, axis=1)
        return X

    

class CollinearColsRemover(BaseEstimator, TransformerMixin):

    def __init__(self, thresh, label_col) -> None:
        self.thresh = thresh
        self.label_col = label_col
    
    
    def fit(self, X:pd.DataFrame, y=None):
        X = X.copy()
        X.drop(labels=self.label_col, axis=1, inplace=True)
        self.cols_to_drop = self._get_collinear_cols(df=X, thresh=self.thresh)
        return self
    
    def transform(self, X:pd.DataFrame):
        n_cols = X.shape[1]
        X = X.drop(self.cols_to_drop, axis=1)
        new_n_cols = X.shape[1]
        n_cols_dropped = n_cols - new_n_cols
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

class ColumnsOrdinalEncoder(ColumnTransformer):

    def __init__(self, col_names, convert_to_int=True) -> None:
        self.col_names = col_names
        self.convert_to_int = convert_to_int
        transformer = ('categorical_encoding', 
                       OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), 
                       self.col_names)
        super().__init__(transformers=[transformer], remainder='passthrough', verbose_feature_names_out=False)

    def fit(self, X, y= None):
        return super().fit(X, y)
    
    def transform(self, X):
        result_array = super().transform(X)

        col_names = self.get_feature_names_out()
        X = pd.DataFrame(result_array, columns=col_names)
        if self.convert_to_int:
            X[self.col_names] = X[self.col_names].astype('int')

        logging.getLogger(self.__class__.__name__).info(f'transformed categorical colums:{self.col_names}')
        return X
    
class BinarizeCol(BaseEstimator, TransformerMixin):

    def __init__(self, col_name, true_val) -> None:
        self.col_name = col_name
        self.true_val = true_val

    def fit(self, X, y=None):
        return self
    
    def transform(self, X:pd.DataFrame):
        X[self.col_name] = np.where(X[self.col_name]==self.true_val, 1, 0)
        logging.getLogger(self.__class__.__name__).info(f'bianrized {self.col_name}')
        return X

class XyDataSplitter(BaseEstimator, TransformerMixin):

    def __init__(self, label_col_name:str) -> None:
        self.label_col_name = label_col_name

    def fit(self, X, y=None):
        return self
    
    def transform(self, Xy:pd.DataFrame):
        X = Xy.drop(labels=self.label_col_name, axis=1)
        y = Xy[self.label_col_name]
        logging.getLogger(self.__class__.__name__).info(f'split data into X and y')
        return X, y


