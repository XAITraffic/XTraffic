"""
Keel datasets (http://sci2s.ugr.es/keel).

@author: David Diaz Vico
@license: MIT
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from types import MappingProxyType
from typing import (
    AbstractSet,
    Any,
    Final,
    Iterator,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
)
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.utils import Bunch

from .base import fetch_file

BASE_URL = 'http://sci2s.ugr.es/keel'
COLLECTIONS: Final = frozenset((
    'classification',
    'missing',
    'imbalanced',
    'multiInstance',
    'multilabel',
    'textClassification',
    'classNoise',
    'attributeNoise',
    'semisupervised',
    'regression',
    'timeseries',
    'unsupervised',
    'lowQuality',
))


# WTFs
IMBALANCED_URLS: Final = (
    'keel-dataset/datasets/imbalanced/imb_IRhigherThan9',
    'keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1',
    'keel-dataset/datasets/imbalanced/imb_IRhigherThan9p2',
    'keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3',
    'dataset/data/imbalanced',
    'keel-dataset/datasets/imbalanced/imb_noisyBordExamples',
    'keel-dataset/datasets/imbalanced/preprocessed',
)

IRREGULAR_DESCR_IMBALANCED_URLS: Final = (
    'keel-dataset/datasets/imbalanced/imb_IRhigherThan9',
    'keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1',
    'keel-dataset/datasets/imbalanced/imb_IRhigherThan9p2',
    'keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3',
)

INCORRECT_DESCR_IMBALANCED_URLS: Final = MappingProxyType(
    {'semisupervised': 'classification'},
)


class KeelOuterCV(object):
    """Iterable over already separated CV partitions of the dataset."""

    def __init__(
        self,
        Xs: Sequence[np.typing.NDArray[float]],
        ys: Sequence[np.typing.NDArray[Union[int, float]]],
        Xs_test: Sequence[np.typing.NDArray[float]],
        ys_test: Sequence[np.typing.NDArray[Union[int, float]]],
    ) -> None:
        self.Xs = Xs
        self.ys = ys
        self.Xs_test = Xs_test
        self.ys_test = ys_test

    def __iter__(self) -> Iterator[Tuple[
        np.typing.NDArray[float],
        np.typing.NDArray[Union[int, float]],
        np.typing.NDArray[float],
        np.typing.NDArray[Union[int, float]],
    ]]:
        return zip(self.Xs, self.ys, self.Xs_test, self.ys_test)


def _load_Xy(
    zipfile: Path,
    csvfile: str,
    sep: str = ',',
    header: Optional[int] = None,
    engine: str = 'python',
    na_values: AbstractSet[str] = frozenset(('?')),
    **kwargs: Any,
) -> Tuple[np.typing.NDArray[float], np.typing.NDArray[Union[int, float]]]:
    """Load a zipped csv file with target in the last column."""
    with ZipFile(zipfile) as z:
        with z.open(csvfile) as c:
            s = io.StringIO(c.read().decode(encoding="utf8"))
            data = pd.read_csv(
                s,
                sep=sep,
                header=header,
                engine=engine,
                na_values=na_values,
                **kwargs,
            )
            data.columns = data.columns.astype(str)
            X = pd.get_dummies(data.iloc[:, :-1])
            y = pd.factorize(data.iloc[:, -1].tolist(), sort=True)[0]
            return X, y


def _load_descr(
    collection: str,
    name: str,
    data_home: Optional[str] = None,
) -> Tuple[int, str]:
    """Load a dataset description."""
    subfolder = os.path.join('keel', collection)
    filename = name + '-names.txt'
    if collection == 'imbalanced':
        for url in IMBALANCED_URLS:
            if url in IRREGULAR_DESCR_IMBALANCED_URLS:
                url = BASE_URL + '/' + url + '/' + 'names' + '/' + filename
            else:
                url = BASE_URL + '/' + url + '/' + filename
            try:
                f = fetch_file(
                    dataname=name,
                    urlname=url,
                    subfolder=subfolder,
                    data_home=data_home,
                )
                break
            except Exception:
                pass
    else:
        collection = (
            INCORRECT_DESCR_IMBALANCED_URLS[collection]
            if collection in INCORRECT_DESCR_IMBALANCED_URLS
            else collection
        )
        url = f"{BASE_URL}/dataset/data/{collection}/{filename}"
        f = fetch_file(
            dataname=name,
            urlname=url,
            subfolder=subfolder,
            data_home=data_home,
        )
    with open(f) as rst_file:
        fdescr = rst_file.read()
        nattrs = fdescr.count("@attribute")
    return nattrs, fdescr


def _fetch_keel_zip(
    collection: str,
    name: str,
    filename: str,
    data_home: Optional[str] = None,
) -> Path:
    """Fetch Keel dataset zip file."""
    subfolder = os.path.join('keel', collection)
    if collection == 'imbalanced':
        for url in IMBALANCED_URLS:
            url = BASE_URL + '/' + url + '/' + filename
            try:
                return fetch_file(
                    dataname=name,
                    urlname=url,
                    subfolder=subfolder,
                    data_home=data_home,
                )
            except Exception:
                pass
    else:
        url = f"{BASE_URL}/dataset/data/{collection}/{filename}"
        return fetch_file(
            dataname=name,
            urlname=url,
            subfolder=subfolder,
            data_home=data_home,
        )
    raise ValueError("Dataset not found")


def _load_folds(
    collection: str,
    name: str,
    nfolds: Literal[None, 1, 5, 10],
    dobscv: bool,
    nattrs: int,
    data_home: Optional[str] = None,
) -> Tuple[
    np.typing.NDArray[float],
    np.typing.NDArray[Union[int, float]],
    Optional[KeelOuterCV],
]:
    """Load a dataset folds."""
    filename = name + '.zip'
    f = _fetch_keel_zip(collection, name, filename, data_home=data_home)
    X, y = _load_Xy(f, name + '.dat', skiprows=nattrs + 4)
    cv = None
    if nfolds in (5, 10):
        fold = 'dobscv' if dobscv else 'fold'
        filename = name + '-' + str(nfolds) + '-' + fold + '.zip'
        f = _fetch_keel_zip(collection, name, filename, data_home=data_home)
        Xs = []
        ys = []
        Xs_test = []
        ys_test = []
        for i in range(nfolds):
            if dobscv:
                # Zipfiles always use fordward slashes, even in Windows.
                _name = f"{name}/{name}-{nfolds}dobscv-{i + 1}"
            else:
                _name = f"{name}-{nfolds}-{i + 1}"
            X_fold, y_fold = _load_Xy(
                f, _name + 'tra.dat', skiprows=nattrs + 4)
            X_test_fold, y_test_fold = _load_Xy(
                f,
                _name + 'tst.dat',
                skiprows=nattrs + 4,
            )
            Xs.append(X_fold)
            ys.append(y_fold)
            Xs_test.append(X_test_fold)
            ys_test.append(y_test_fold)

        cv = KeelOuterCV(Xs, ys, Xs_test, ys_test)
    return X, y, cv


@overload
def fetch(
    collection: str,
    name: str,
    data_home: Optional[str] = None,
    nfolds: Literal[None, 1, 5, 10] = None,
    dobscv: bool = False,
    *,
    return_X_y: Literal[False] = False,
) -> Bunch:
    pass


@overload
def fetch(
    collection: str,
    name: str,
    data_home: Optional[str] = None,
    nfolds: Literal[None, 1, 5, 10] = None,
    dobscv: bool = False,
    *,
    return_X_y: Literal[True],
) -> Tuple[np.typing.NDArray[float], np.typing.NDArray[Union[int, float]]]:
    pass


def fetch(
    collection: str,
    name: str,
    data_home: Optional[str] = None,
    nfolds: Literal[None, 1, 5, 10] = None,
    dobscv: bool = False,
    *,
    return_X_y: bool = False,
) -> Union[
    Bunch,
    Tuple[np.typing.NDArray[float], np.typing.NDArray[Union[int, float]]],
]:
    """
    Fetch Keel dataset.

    Fetch a Keel dataset by collection and name. More info at
    http://sci2s.ugr.es/keel.

    Parameters
    ----------
    collection : string
        Collection name.
    name : string
        Dataset name.
    data_home : string or None, default None
        Specify another download and cache folder for the data sets. By default
        all scikit-learn data is stored in ‘~/scikit_learn_data’ subfolders.
    nfolds : int, default=None
        Number of folds. Depending on the dataset, valid values are
        {None, 1, 5, 10}.
    dobscv : bool, default=False
        If folds are in {5, 10}, indicates that the cv folds are distribution
        optimally balanced stratified. Only available for some datasets.
    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object.
    kwargs : dict
        Optional key-value arguments

    Returns
    -------
    data : Bunch
        Dictionary-like object with all the data and metadata.

    (data, target) : tuple if ``return_X_y`` is True

    """
    if collection not in COLLECTIONS:
        raise ValueError('Avaliable collections are ' + str(list(COLLECTIONS)))
    nattrs, DESCR = _load_descr(collection, name, data_home=data_home)
    X, y, cv = _load_folds(
        collection,
        name,
        nfolds,
        dobscv,
        nattrs,
        data_home=data_home,
    )

    if return_X_y:
        return X, y

    return Bunch(
        data=X,
        target=y,
        train_indices=[],
        validation_indices=[],
        test_indices=[],
        inner_cv=None,
        outer_cv=cv,
        DESCR=DESCR,
    )
