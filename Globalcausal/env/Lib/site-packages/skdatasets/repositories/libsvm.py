"""
LIBSVM datasets (https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets).

@author: David Diaz Vico
@license: MIT
"""
from __future__ import annotations

import os
import sys
from typing import Final, Literal, Sequence, Tuple, overload

import numpy as np
import scipy as sp
from sklearn.datasets import load_svmlight_file, load_svmlight_files
from sklearn.model_selection import PredefinedSplit
from sklearn.utils import Bunch

from .base import DatasetNotFoundError, fetch_file

BASE_URL: Final = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets'
COLLECTIONS: Final = frozenset((
    'binary',
    'multiclass',
    'regression',
    'string',
))


def _fetch_partition(
    collection: str,
    name: str,
    partition: str,
    data_home: str | None = None,
) -> str | None:
    """Fetch dataset partition."""
    subfolder = os.path.join('libsvm', collection)
    dataname = name.replace('/', '-')

    url = f"{BASE_URL}/{collection}/{name}{partition}"

    for data_url in (f"{url}.bz2", url):
        try:
            return os.fspath(
                fetch_file(
                    dataname,
                    urlname=data_url,
                    subfolder=subfolder,
                    data_home=data_home,
                ),
            )
        except DatasetNotFoundError:
            pass

    return None


def _load(
    collection: str,
    name: str,
    data_home: str | None = None,
) -> Tuple[
    np.typing.NDArray[float],
    np.typing.NDArray[int | float],
    Sequence[int],
    Sequence[int],
    Sequence[int],
    PredefinedSplit,
]:
    """Load dataset."""
    filename = _fetch_partition(collection, name, '', data_home)
    filename_tr = _fetch_partition(collection, name, '.tr', data_home)
    filename_val = _fetch_partition(collection, name, '.val', data_home)
    filename_t = _fetch_partition(collection, name, '.t', data_home)
    filename_r = _fetch_partition(collection, name, '.r', data_home)

    if (filename_tr is not None) and (filename_val is not None) and (filename_t is not None):

        _, _, X_tr, y_tr, X_val, y_val, X_test, y_test = load_svmlight_files([
            filename,
            filename_tr,
            filename_val,
            filename_t,
        ])

        cv = PredefinedSplit([-1] * X_tr.shape[0] + [0] * X_val.shape[0])

        X = sp.sparse.vstack((X_tr, X_val, X_test))
        y = np.hstack((y_tr, y_val, y_test))

        # Compute indices
        train_indices = list(range(X_tr.shape[0]))
        validation_indices = list(range(
            X_tr.shape[0],
            X_tr.shape[0] + X_val.shape[0],
        ))
        test_indices = list(range(X_tr.shape[0] + X_val.shape[0], X.shape[0]))

    elif (filename_tr is not None) and (filename_val is not None):

        _, _, X_tr, y_tr, X_val, y_val = load_svmlight_files([
            filename,
            filename_tr,
            filename_val,
        ])

        cv = PredefinedSplit([-1] * X_tr.shape[0] + [0] * X_val.shape[0])

        X = sp.sparse.vstack((X_tr, X_val))
        y = np.hstack((y_tr, y_val))

        # Compute indices
        train_indices = list(range(X_tr.shape[0]))
        validation_indices = list(range(X_tr.shape[0], X.shape[0]))
        test_indices = []

    elif (filename_t is not None) and (filename_r is not None):

        X_tr, y_tr, X_test, y_test, X_remaining, y_remaining = (
            load_svmlight_files([
                filename,
                filename_t,
                filename_r,
            ])
        )

        X = sp.sparse.vstack((X_tr, X_test, X_remaining))
        y = np.hstack((y_tr, y_test, y_remaining))

        # Compute indices
        train_indices = list(range(X_tr.shape[0]))
        validation_indices = []
        test_indices = list(
            range(
                X_tr.shape[0], X_tr.shape[0] + X_test.shape[0],
            ),
        )

        cv = None

    elif filename_t is not None:

        X_tr, y_tr, X_test, y_test = load_svmlight_files([
            filename,
            filename_t,
        ])

        X = sp.sparse.vstack((X_tr, X_test))
        y = np.hstack((y_tr, y_test))

        # Compute indices
        train_indices = list(range(X_tr.shape[0]))
        validation_indices = []
        test_indices = list(range(X_tr.shape[0], X.shape[0]))

        cv = None

    else:

        X, y = load_svmlight_file(filename)

        # Compute indices
        train_indices = []
        validation_indices = []
        test_indices = []

        cv = None

    return X, y, train_indices, validation_indices, test_indices, cv


@overload
def fetch(
    collection: str,
    name: str,
    *,
    data_home: str | None = None,
    return_X_y: Literal[False] = False,
) -> Bunch:
    pass


@overload
def fetch(
    collection: str,
    name: str,
    *,
    data_home: str | None = None,
    return_X_y: Literal[True],
) -> Tuple[np.typing.NDArray[float], np.typing.NDArray[int | float]]:
    pass


def fetch(
    collection: str,
    name: str,
    *,
    data_home: str | None = None,
    return_X_y: bool = False,
) -> Bunch | Tuple[np.typing.NDArray[float], np.typing.NDArray[int | float]]:
    """
    Fetch LIBSVM dataset.

    Fetch a LIBSVM dataset by collection and name. More info at
    https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets.

    Parameters
    ----------
    collection : string
        Collection name.
    name : string
        Dataset name.
    data_home : string or None, default None
        Specify another download and cache folder for the data sets. By default
        all scikit-learn data is stored in ‘~/scikit_learn_data’ subfolders.
    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object.

    Returns
    -------
    data : Bunch
        Dictionary-like object with all the data and metadata.

    (data, target) : tuple if ``return_X_y`` is True

    """
    if collection not in COLLECTIONS:
        raise Exception('Avaliable collections are ' + str(list(COLLECTIONS)))

    X, y, train_indices, validation_indices, test_indices, cv = _load(
        collection,
        name,
        data_home=data_home,
    )

    if return_X_y:
        return X, y

    return Bunch(
        data=X,
        target=y,
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
        inner_cv=cv,
        outer_cv=None,
        DESCR=name,
    )
