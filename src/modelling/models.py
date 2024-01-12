"""Scripts for constituting models"""
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import numpy as np
import optuna
from abc import ABC, abstractmethod
import configparser
import logging

from src.utils import utils


class WrappedModel(ABC):
    @abstractmethod
    def train(self, X_train, y_train, transform_pipeline: Pipeline = None):
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
    def objective_function(self, trial, X, y, transform_pipeline: Pipeline = None):
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
        self.n_jobs = int(n_jobs)
        self.model = LogisticRegression
        self.is_tuned = False
        self.tuned_params = None

    def objective_function(self, trial, X, y, transform_pipeline: Pipeline):
        C = trial.suggest_float("C", 0.1, 10, log=True)
        penalty = trial.suggest_categorical("penalty", ["l1", "l2"])

        model = self.model(C=C, penalty=penalty, solver=self.solver, n_jobs=self.n_jobs)

        model_pipeline = _make_model_pipeline(
            transform_pipeline=transform_pipeline, model=model
        )
        scores = cross_val_score(
            model_pipeline, X, y, cv=5, scoring="roc_auc", error_score="raise"
        )
        roc_auc = np.mean(scores)

        print(f"Trial {trial.number}, C: {C}, penalty: {penalty}, ROC-AUC: {roc_auc}")
        return roc_auc

    def init_model(self):
        if self.tuned_params is not None:
            print("using tuned params")
            model = self.model(
                **self.tuned_params, solver=self.solver, n_jobs=self.n_jobs
            )
        else:
            print("not using tuned")
            model = self.model(solver=self.solver, n_jobs=self.n_jobs)

        self.model = model

    def train(self, X, y, transform_pipeline: Pipeline = None):
        if not self.is_tuned:
            self.init_model()

        logging.getLogger(self.__class__.__name__).info(
            f"training with {type(self.model)}"
        )
        model_pipeline = _make_model_pipeline(
            transform_pipeline=transform_pipeline, model=self.model
        )
        model_pipeline.fit(X, y)

        return model_pipeline

    def predict(self, X, return_prob=False):
        if return_prob:
            y_pred = self.model.predict_proba(X)[:, 1]
        else:
            y_pred = self.model.predict(X)
        return y_pred


class RandomForestWrapper(WrappedModel):
    def __init__(self, n_jobs) -> None:
        self.n_jobs = int(n_jobs)
        self.model = RandomForestClassifier
        self.is_tuned = False
        self.tuned_params = None

    def objective_function(self, trial, X, y, transform_pipeline: Pipeline):
        n_estimators = trial.suggest_int("n_estimators", 2, 150)
        max_depth = trial.suggest_int("max_depth", 1, 50)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 15)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 15)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            n_jobs=self.n_jobs,
        )
        model_pipeline = _make_model_pipeline(
            transform_pipeline=transform_pipeline, model=model
        )
        scores = cross_val_score(model_pipeline, X, y, cv=5, scoring="roc_auc")
        roc_auc = np.mean(scores)
        print(
            f"Trial {trial.number}, n_estimators: {n_estimators}, max_depth: {max_depth}, "
            f"min_samples_split: {min_samples_split}, min_samples_leaf: {min_samples_leaf}, ROC-AUC: {roc_auc}"
        )
        return roc_auc

    def init_model(self):
        if self.tuned_params is not None:
            model = self.model(**self.tuned_params, n_jobs=self.n_jobs)
        else:
            model = self.model(n_jobs=self.n_jobs)

        self.model = model

    def train(self, X, y, transform_pipeline: Pipeline = None):
        if not self.is_tuned:
            self.init_model()

        logging.getLogger(self.__class__.__name__).info(
            f"training with {type(self.model)}"
        )
        model_pipeline = _make_model_pipeline(
            transform_pipeline=transform_pipeline, model=self.model
        )
        model_pipeline.fit(X, y)
        return model_pipeline

    def predict(self, X, return_prob=False):
        if return_prob:
            y_pred = self.model.predict_proba(X)[:, 1]
        else:
            y_pred = self.model.predict(X)
        return y_pred


class XgboostWrapper(WrappedModel):
    def __init__(self, n_jobs) -> None:
        self.n_jobs = int(n_jobs)
        self.use_label_encoder = False
        self.model = XGBClassifier
        self.is_tuned = False
        self.tuned_params = None

    def objective_function(self, trial, X, y, transform_pipeline: Pipeline):
        n_estimators = trial.suggest_int("n_estimators", 2, 150)
        max_depth = trial.suggest_int("max_depth", 1, 50)
        learning_rate = trial.suggest_float("learning_rate", 0.001, 0.9, log=True)
        min_child_weight = trial.suggest_int("min_child_weight", 1, 10)
        subsample = trial.suggest_float("subsample", 0.5, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
        gamma = trial.suggest_float("gamma", 0, 1.0)
        reg_alpha = trial.suggest_float("reg_alpha", 0, 1)
        reg_lambda = trial.suggest_float("reg_lambda", 0, 1)

        model = self.model(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            use_label_encoder=self.use_label_encoder,
            n_jobs=self.n_jobs,
        )
        model_pipeline = _make_model_pipeline(
            transform_pipeline=transform_pipeline, model=model
        )
        # Using cross_val_score to get the average ROC-AUC score for each fold
        scores = cross_val_score(model_pipeline, X, y, cv=5, scoring="roc_auc")
        roc_auc = np.mean(scores)
        # Printing intermediate results
        print(
            f"Trial {trial.number}, n_estimators: {n_estimators}, max_depth: {max_depth}, learning_rate: {learning_rate},"
            f"min_child_weight: {min_child_weight}, subsample: {subsample}, colsample_bytree: {colsample_bytree}, "
            f"gamma: {gamma}, reg_alpha: {reg_alpha}, reg_lambda: {reg_lambda}, ROC-AUC: {roc_auc}"
        )
        return roc_auc

    def init_model(self):
        if self.tuned_params is not None:
            model = self.model(**self.tuned_params, n_jobs=self.n_jobs)
        else:
            model = self.model(
                n_jobs=self.n_jobs, use_labael_encoder=self.use_label_encoder
            )

        self.model = model

    def train(self, X, y, transform_pipeline: Pipeline = None):
        if not self.is_tuned:
            self.init_model()

        logging.getLogger(self.__class__.__name__).info(
            f"training with {type(self.model)}"
        )

        model_pipeline = _make_model_pipeline(
            transform_pipeline=transform_pipeline, model=self.model
        )
        model_pipeline.fit(X, y)
        return model_pipeline

    def predict(self, X, return_prob=False):
        if return_prob:
            y_pred = self.model.predict_proba(X)[:, 1]
        else:
            y_pred = self.model.predict(X)
        return y_pred


def _make_model_pipeline(transform_pipeline, model):
    model_pipeline = sklearn.clone(transform_pipeline)
    model_pipeline.steps.append(["model", model])
    return model_pipeline
