"""utilities for training model with training data"""

import pandas as pd
import configparser
import logging
import pathlib

from src.utils import utils

logger = logging.getLogger(__name__)
# proj_root = utils.get_proj_root()

# config = configparser.ConfigParser(interpolation=None)
# config.read(proj_root.joinpath('config/data_config.ini'))

def get_training_data(file_path:pathlib.Path=None):

    df = pd.read_csv(file_path)

    return df


def train_test_split(df:pd.DataFrame, final_year:int, save_data=False):

    # final_year = int(config['year_limits']['end_year'])

    training_data = df.loc[df['year'] != final_year]
    testing_data = df.loc[df['year'] == final_year]

    logger.info(f"data split into: training ({training_data.shape}) and test ({testing_data.shape}) sets ")


    return training_data, testing_data



def split_Xy(df:pd.DataFrame, label_col_name:str):

    X = df.drop(label_col_name, axis=1)
    y = df[label_col_name]

    return X, y

def save_data(data:pd.DataFrame, path:pathlib.Path):

    data.to_csv(path_or_buf=path, index=False)