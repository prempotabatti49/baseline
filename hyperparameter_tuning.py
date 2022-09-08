# This module will help in replicating hyperparameter tuning quickly. We have taken iris data for example
# There are three types of tuning in this module:
# 1. Gridsearch
# 2. RandomSearch
# 3. Hyperopt

from dataclasses import dataclass
from functools import partial

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from hyperopt import hp, fmin, Trials, tpe
from hyperopt.pyll.base import scope

if __name__ == "__main__":
    data = load_iris()
    X = data.data
    y = data.target

    ##GridsearchCV

    scl = StandardScaler()
    pca = PCA()
    rf = RandomForestClassifier()

    classifier = pipeline.Pipeline([
        ("scl", scl),
        ("pca", pca),
        ("rf", rf)]
    )
    param_grid = {
        "pca__n_components": [1, 2, 3, 4],
        "rf__n_estimators": [100, 200, 300, 20, 30],
        "rf__max_depth": [1, 3, 5, 10],
        "rf__criterion": ["gini", "entropy"],
    }


    model = GridSearchCV(
        estimator=classifier,
        param_grid=param_grid,
        scoring="accuracy",
        verbose=10,
        n_jobs=1,
        cv=5,
    )
    model.fit(X, y)
    print("--------------------")
    print(model.best_score_)
    print("--------------------")
    print(model.best_estimator_.get_params())
    print("--------------------")

    ##Hyperopt

    @dataclass
    class Dataset:
        X: np.ndarray
        y: np.ndarray


    def objective_function(X, y, params_):
        clf = train_model(params_, X, y)
        loss = estimate_loss(clf, X, y)
        return loss


    def train_model(params_, X, y):
        rf = RandomForestClassifier(**params_)
        rf.fit(X, y)
        return rf


    def estimate_loss(clf, X, y):
        y_pred = clf.predict(X)
        loss = recall_score(y, y_pred, average="macro")
        return loss


    param_grid = {
        "n_estimators": scope.int(hp.quniform("n_estimators", 100, 600, 1)),
        "max_depth": scope.int(hp.quniform("max_depth", 3, 15, 1)),
        "criterion": hp.choice("criterion", ["gini", "entropy"]),
    }
    dataset = Dataset(X=X, y=y)
    objective = partial(
        objective_function,
        dataset.X,
        dataset.y
    )
    trials = Trials()
    params = fmin(
        fn=objective,
        space=param_grid,
        algo=tpe.suggest,
        trials=trials,
        max_evals=15,
    )
    print("--------------------")
    print("--------------------")
    print(params)

