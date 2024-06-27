"""
@author: David Diaz Vico
@license: MIT
"""
from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING, Iterable, Tuple, Union

import numpy as np
from sacred.observers import FileStorageObserver
from sklearn.datasets import load_diabetes, load_iris, load_wine
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import Bunch

from skdatasets.utils.experiment import (
    create_experiments,
    experiment,
    fetch_scores,
    run_experiments,
)

if TYPE_CHECKING:
    from skdatasets.utils.experiment import CVLike

    ExplicitSplitType = Tuple[
        np.typing.NDArray[float],
        np.typing.NDArray[Union[float, int]],
        np.typing.NDArray[float],
        np.typing.NDArray[Union[float, int]],
    ]


def _dataset(
    inner_cv: CVLike = None,
    outer_cv: CVLike = None,
) -> Bunch:
    data = load_diabetes()
    if outer_cv is None:
        X, X_test, y, y_test = train_test_split(data.data, data.target)
        data.data = X
        data.target = y
        data.data_test = X_test
        data.target_test = y_test
        data.outer_cv = None
    else:
        data.data_test = data.target_test = None
        data.outer_cv = outer_cv
    data.inner_cv = inner_cv

    return data


def _estimator(cv: CVLike) -> GridSearchCV:
    return GridSearchCV(
        DecisionTreeRegressor(),
        {'max_depth': [2, 4]},
        cv=cv,
    )


def _experiment(
    inner_cv: CVLike,
    outer_cv: CVLike | Iterable[ExplicitSplitType],
) -> None:
    with tempfile.TemporaryDirectory() as tmpdirname:
        e = experiment(_dataset, _estimator)
        e.observers.append(FileStorageObserver(tmpdirname))
        e.run(
            config_updates={
                'dataset': {
                    'inner_cv': inner_cv,
                    'outer_cv': outer_cv,
                },
            },
        )


def test_nested_cv() -> None:
    """Tests nested CV experiment."""
    _experiment(3, 3)


def test_inner_cv() -> None:
    """Tests inner CV experiment."""
    _experiment(3, None)


def test_explicit_inner_folds() -> None:
    """Tests explicit inner folds experiment."""
    X, y = load_diabetes(return_X_y=True)
    _experiment(
        [
            (np.arange(10), np.arange(10, 20)),
            (np.arange(10, 20), np.arange(20, 30)),
            (np.arange(20, 30), np.arange(30, 40)),
        ],
        3,
    )


def test_explicit_outer_folds_indexes() -> None:
    """Tests explicit outer folds experiment."""
    X, y = load_diabetes(return_X_y=True)
    _experiment(
        3,
        [
            (np.arange(10), np.arange(10, 20)),
            (np.arange(10, 20), np.arange(20, 30)),
            (np.arange(20, 30), np.arange(30, 40)),
        ],
    )


def test_explicit_outer_folds() -> None:
    """Tests explicit outer folds experiment."""
    X, y = load_diabetes(return_X_y=True)
    _experiment(
        3,
        [
            (X[:10], y[:10], X[10:20], y[10:20]),
            (X[10:20], y[10:20], X[20:30], y[20:30]),
            (X[20:30], y[20:30], X[30:40], y[30:40]),
        ],
    )


def test_explicit_nested_folds() -> None:
    """Tests explicit nested folds experiment."""
    X, y = load_diabetes(return_X_y=True)
    _experiment(
        [
            (np.arange(3, 10), np.arange(3)),
            (np.concatenate((np.arange(3), np.arange(7, 10))), np.arange(3, 7)),
            (np.arange(7, 10), np.arange(7)),
        ],
        [
            (np.arange(10), np.arange(10, 20)),
            (np.arange(10, 20), np.arange(20, 30)),
            (np.arange(20, 30), np.arange(30, 40)),
        ],
    )


def test_create_experiments_basic() -> None:

    with tempfile.TemporaryDirectory() as tmpdirname:
        experiments = create_experiments(
            estimators={
                "knn-3": KNeighborsClassifier(n_neighbors=3),
                "knn-5": KNeighborsClassifier(n_neighbors=5),
                "knn-7": KNeighborsClassifier(n_neighbors=7),
            },
            datasets={
                "iris": load_iris(),
                "wine": load_wine(),
            },
            storage=tmpdirname,
        )

        ids = run_experiments(experiments)

        scores = fetch_scores(
            storage=tmpdirname,
            ids=ids,
        )

        assert scores.dataset_names == ("iris", "wine")
        assert scores.estimator_names == ("knn-3", "knn-5", "knn-7")
        np.testing.assert_allclose(
            scores.scores_mean,
            [
                [0.96666667, 0.97333333, 0.98],
                [0.70285714, 0.69126984, 0.68063492],
            ],
        )
        np.testing.assert_allclose(
            scores.scores_std,
            [
                [0.02108185, 0.02494438, 0.01632993],
                [0.07920396, 0.04877951, 0.0662983],
            ],
            rtol=1e-6,
        )
