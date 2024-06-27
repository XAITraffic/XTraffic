"""
Keras datasets (https://keras.io/datasets).

@author: David Diaz Vico
@license: MIT
"""

from __future__ import annotations

from typing import Any, Literal, Tuple, overload

import numpy as np
from sklearn.utils import Bunch

from tensorflow.keras.datasets import (
    boston_housing,
    cifar10,
    cifar100,
    fashion_mnist,
    imdb,
    mnist,
    reuters,
)

DATASETS = {
    'boston_housing': boston_housing.load_data,
    'cifar10': cifar10.load_data,
    'cifar100': cifar100.load_data,
    'fashion_mnist': fashion_mnist.load_data,
    'imdb': imdb.load_data,
    'mnist': mnist.load_data,
    'reuters': reuters.load_data,
}


@overload
def fetch(
    name: str,
    *,
    return_X_y: Literal[False] = False,
    **kwargs: Any,
) -> Bunch:
    pass


@overload
def fetch(
    name: str,
    *,
    return_X_y: Literal[True],
    **kwargs: Any,
) -> Tuple[np.typing.NDArray[float], np.typing.NDArray[int]]:
    pass


def fetch(
    name: str,
    *,
    return_X_y: bool = False,
    **kwargs: Any,
) -> Bunch | Tuple[np.typing.NDArray[float], np.typing.NDArray[int]]:
    """
    Fetch Keras dataset.

    Fetch a Keras dataset by name. More info at https://keras.io/datasets.

    Parameters
    ----------
    name : string
        Dataset name.
    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object.
    **kwargs : dict
        Optional key-value arguments. See https://keras.io/datasets.

    Returns
    -------
    data : Bunch
        Dictionary-like object with all the data and metadata.

    (data, target) : tuple if ``return_X_y`` is True

    """
    (X_train, y_train), (X_test, y_test) = DATASETS[name](**kwargs)
    if len(X_train.shape) > 2:
        name = name + ' ' + str(X_train.shape[1:]) + ' shaped'
        X_max = np.iinfo(X_train[0][0].dtype).max
        n_features = np.prod(X_train.shape[1:])
        X_train = X_train.reshape([X_train.shape[0], n_features]) / X_max
        X_test = X_test.reshape([X_test.shape[0], n_features]) / X_max

    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))

    if return_X_y:
        return X, y

    return Bunch(
        data=X,
        target=y,
        train_indices=list(range(len(X_train))),
        validation_indices=[],
        test_indices=list(range(len(X_train), len(X))),
        inner_cv=None,
        outer_cv=None,
        DESCR=name,
    )
