"""Scripts for constituting models"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import optuna
from abc import ABC, abstractmethod
import configparser

from src.utils import utils


def objective_function(trial, X, y):
    C = trial.suggest_float('C', 0.1, 10, log=True)
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])

    model = LogisticRegression(
        C=C,
        penalty=penalty,
        solver='liblinear',
        n_jobs=-1
    )

    # Using cross_val_score to get the average precision score for each fold
    scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    roc_auc = np.mean(scores)
    # Printing intermediate results
    print(f"Trial {trial.number}, C: {C}, penalty: {penalty}, ROC-AUC: {roc_auc}")
    return roc_auc

# proj_root = utils.get_proj_root()
# config = configparser.ConfigParser(interpolation=None)
# config.read(proj_root.joinpath('config/data_config.ini'))



class WrappedModel(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        """
        Train the model on the given training data.

        Parameters:
        - X_train: Input features for training.
        - y_train: Target labels for training.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Make predictions using the trained model.

        Parameters:
        - X: Input features for prediction.

        Returns:
        - Predicted labels.
        """
        pass

    @abstractmethod
    def objective_function(self, trial, X, y):
        """
        Objective function for optimization tasks.

        Parameters:
        - X: Input features for objective evaluation.
        - y: Target labels for objective evaluation.

        Returns:
        - Objective value (e.g., accuracy, loss).
        """
        pass

    @abstractmethod
    def init_model(self, params):

        pass


class LogisticWrapper(WrappedModel):
    
    def __init__(self, solver, n_jobs) -> None:
        self.solver = solver
        self.n_jobs=int(n_jobs)
        self.model = LogisticRegression
        self.is_tuned=False
        self.tunable_params =('C', 'penalty')
        self.constant_params = ('solver', 'n_jobs')
        self.tuned_params=None


    def objective_function(self, trial, X, y):
        C = trial.suggest_float('C', 0.1, 10, log=True)
        penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])

        model = self.model(
            C=C,
            penalty=penalty,
            solver=self.solver,
            n_jobs=self.n_jobs
        )

        # Using cross_val_score to get the average precision score for each fold
        scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
        roc_auc = np.mean(scores)
        # Printing intermediate results
        print(f"Trial {trial.number}, C: {C}, penalty: {penalty}, ROC-AUC: {roc_auc}")
        return roc_auc
    
    def init_model(self):

        if self.tuned_params is not None:
            model = self.model(**self.tuned_params, 
                               solver=self.solver, n_jobs=self.n_jobs)
        else:
            model = self.model(solver=self.solver, n_jobs=self.n_jobs)

        self.model = model

        
    def train(self, X, y):

        if not self.is_tuned:
            self.init_model()

        print(type(self.model))
        self.model.fit(X, y)
        return self.model

    def predict(self, X, return_prob=False):
        if return_prob:
            y_pred = self.model.predict_proba(X)[:, 1]
        else:
            y_pred = self.model.predict(X)
        return y_pred

    
class RandomForestWrapper(WrappedModel):
    
    def __init__(self, n_jobs) -> None:
        self.n_jobs=int(n_jobs)
        self.model = RandomForestClassifier
        self.is_tuned=False
        self.tuned_params=None


    def objective_function(self, trial, X, y):
        n_estimators = trial.suggest_int('n_estimators', 2, 150)
        max_depth = trial.suggest_int('max_depth', 1, 50)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 15)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 15)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1
        )

        # Using cross_val_score to get the average ROC-AUC score for each fold
        scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
        roc_auc = np.mean(scores)
        # Printing intermediate results
        print(f"Trial {trial.number}, n_estimators: {n_estimators}, max_depth: {max_depth}, "
            f"min_samples_split: {min_samples_split}, min_samples_leaf: {min_samples_leaf}, ROC-AUC: {roc_auc}")
        return roc_auc

    def init_model(self):

        if self.tuned_params is not None:
            model = self.model(**self.tuned_params, 
                                n_jobs=self.n_jobs)
        else:
            model = self.model(n_jobs=self.n_jobs)

        self.model = model
        
    def train(self, X, y):

        if not self.is_tuned:
            self.init_model()

        print(type(self.model))
        self.model.fit(X, y)
        return self.model

    def predict(self, X, return_prob=False):
        if return_prob:
            y_pred = self.model.predict_proba(X)[:, 1]
        else:
            y_pred = self.model.predict(X)
        return y_pred
    