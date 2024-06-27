"""
UCI datasets (https://archive.ics.uci.edu/ml/datasets.html).

@author: David Diaz Vico
@license: MIT
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal, Optional, Tuple, Union, overload

import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils import Bunch

from .base import fetch_file

BASE_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases'


def _load_csv(
    fname: Path,
    **kwargs: Any,
) -> Tuple[
    np.typing.NDArray[float],
    np.typing.NDArray[Union[float, int, str]],
]:
    """Load a csv with targets in the last column and features in the rest."""
    data = np.genfromtxt(
        fname,
        dtype=str,
        delimiter=',',
        encoding=None,
        **kwargs,
    )
    X = data[:, :-1]
    try:
        X = X.astype(float)
    except ValueError:
        pass

    y = data[:, -1]

    return X, y


def _fetch(
    name: str,
    data_home: Optional[str] = None,
) -> Tuple[
    np.typing.NDArray[float],
    np.typing.NDArray[Union[float, int]],
    Optional[np.typing.NDArray[float]],
    Optional[np.typing.NDArray[Union[float, int]]],
    str,
    np.typing.NDArray[str],
]:
    """Fetch dataset."""
    subfolder = 'uci'
    filename_str = name + '.data'
    url = BASE_URL + '/' + name + '/' + filename_str

    filename = fetch_file(
        dataname=name,
        urlname=url,
        subfolder=subfolder,
        data_home=data_home,
    )
    X, y = _load_csv(filename)
    target_names = None
    ordinal_encoder = OrdinalEncoder(dtype=np.int64)
    if y.dtype.type is np.str_:
        y = ordinal_encoder.fit_transform(y.reshape(-1, 1))[:, 0]
        target_names = ordinal_encoder.categories_[0]
    try:
        filename_str = name + '.test'
        url = BASE_URL + '/' + name + '/' + filename_str
        filename = fetch_file(
            dataname=name,
            urlname=url,
            subfolder=subfolder,
            data_home=data_home,
        )
        X_test: Optional[np.typing.NDArray[float]]
        y_test: Optional[np.typing.NDArray[Union[float, int, str]]]
        X_test, y_test = _load_csv(filename)

        if y.dtype.type is np.str_:
            y_test = ordinal_encoder.transform(y_test.reshape(-1, 1))[:, 0]

    except Exception:
        X_test = None
        y_test = None
    try:
        filename_str = name + '.names'
        url = BASE_URL + '/' + name + '/' + filename_str
        filename = fetch_file(
            dataname=name,
            urlname=url,
            subfolder=subfolder,
            data_home=data_home,
        )
    except Exception:
        filename_str = name + '.info'
        url = BASE_URL + '/' + name + '/' + filename_str
        filename = fetch_file(
            dataname=name,
            urlname=url,
            subfolder=subfolder,
            data_home=data_home,
        )
    with open(filename) as rst_file:
        fdescr = rst_file.read()
    return X, y, X_test, y_test, fdescr, target_names


@overload
def fetch(
    name: str,
    data_home: Optional[str] = None,
    *,
    return_X_y: Literal[False] = False,
) -> Bunch:
    pass


@overload
def fetch(
    name: str,
    data_home: Optional[str] = None,
    *,
    return_X_y: Literal[True],
) -> Tuple[np.typing.NDArray[float], np.typing.NDArray[float]]:
    pass


def fetch(
    name: str,
    data_home: Optional[str] = None,
    *,
    return_X_y: bool = False,
) -> Union[
    Bunch,
    Tuple[np.typing.NDArray[float], np.typing.NDArray[float]],
]:
    """
    Fetch UCI dataset.

    Fetch a UCI dataset by name. More info at
    https://archive.ics.uci.edu/ml/datasets.html.

    Parameters
    ----------
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
    X_train, y_train, X_test, y_test, DESCR, target_names = _fetch(
        name,
        data_home=data_home,
    )

    if X_test is None or y_test is None:
        X = X_train
        y = y_train

        train_indices = None
        test_indices = None
    else:
        X = np.concatenate((X_train, X_test))
        y = np.concatenate((y_train, y_test))

        train_indices = list(range(len(X_train)))
        test_indices = list(range(len(X_train), len(X)))

    if return_X_y:
        return X, y

    return Bunch(
        data=X,
        target=y,
        train_indices=train_indices,
        validation_indices=[],
        test_indices=test_indices,
        inner_cv=None,
        outer_cv=None,
        DESCR=DESCR,
        target_names=target_names,
    )
