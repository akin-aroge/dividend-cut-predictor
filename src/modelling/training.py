"""utilities for training model with training data"""

import pandas as pd
import configparser
import logging
import pathlib
import optuna
from sklearn.metrics import roc_auc_score, classification_report
from src.modelling import models


from src.utils import utils

logger = logging.getLogger(__name__)
# proj_root = utils.get_proj_root()

# config = configparser.ConfigParser(interpolation=None)
# config.read(proj_root.joinpath('config/data_config.ini'))

class ModelTrainer():

    def __init__(self, model_class:models.WrappedModel, 
                 training_data, testing_data, label_col_name, model_output_path) -> None:

        self.direction=None
        self.best_params = None
        self.model_output_path = model_output_path
        self.model_class = model_class
        self.n_trials = 100
        self.best_score = None
        # self.training_data_path = training_data_path
        self.training_data = training_data
        self.testing_data = testing_data
        self.label_col_name = label_col_name

        # self.model=None


    def tune_model(self, n_trials, maximize=True):
        
        if maximize:
            direction='maximize'
        else:
            direction = 'minimize'
        self.direction=direction

        X_train, y_train = split_Xy(self.training_data, label_col_name=self.label_col_name)

        study = optuna.create_study(direction=direction)
        study.optimize(lambda trial: self.model_class.objective_function(
            trial=trial, X=X_train, y=y_train),
            n_trials=n_trials
        )

        self.best_params = study.best_params
        self.best_score = study.best_value
        self.model_class.tuned_params = study.best_params
        # model = self.model_class.init_model(params=self.best_params)
        self.model_class.init_model()
        # self.model_class.model = model
        self.model_class.is_tuned=True
        print(f'best score is: {self.best_score}')

    def train_model(self,  save_model=True):

        X_train, y_train = split_Xy(self.training_data, label_col_name=self.label_col_name)

        model_is_tuned = self.model_class.is_tuned
        if not model_is_tuned:
            print('info: model is not tuned')

        trained_model = self.model_class.train(X=X_train, y=y_train)

        if save_model:
            utils.save_value(trained_model, fname=self.model_output_path)

        return trained_model
    
    def evaluate_model(self, show_report=True):
        print(self.model_class.model.classes_)
        X_test, y_test = split_Xy(self.testing_data, label_col_name=self.label_col_name)

        y_pred_prob = self.model_class.predict(X_test, return_prob=True)
        y_pred = self.model_class.predict(X_test, return_prob=False)
        score = roc_auc_score(y_true=y_test, y_score=y_pred_prob)

        if show_report:
            print(classification_report(y_true=y_test, y_pred=y_pred))
        return score




def get_features(path:pathlib.Path):

    features_list = utils.load_value(path)

    return features_list


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

def get_model_class(model_name:str):
    if model_name == 'logistic_regression':
        model = models.LogisticWrapper
    elif model_name == 'random_forest':
        model = models.RandomForestWrapper

    return model



def save_data(data:pd.DataFrame, path:pathlib.Path):

    data.to_csv(path_or_buf=path, index=False)




