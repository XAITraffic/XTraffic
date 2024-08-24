"""
Gunnar Raetsch benchmark datasets
(https://github.com/tdiethe/gunnar_raetsch_benchmark_datasets).

@author: David Diaz Vico
@license: MIT
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import (
    Final,
    Iterator,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
)

import numpy as np
from scipy.io import loadmat
from sklearn.utils import Bunch

from .base import fetch_file

DATASETS: Final = frozenset((
    'banana',
    'breast_cancer',
    'diabetis',
    'flare_solar',
    'german',
    'heart',
    'image',
    'ringnorm',
    'splice',
    'thyroid',
    'titanic',
    'twonorm',
    'waveform',
))


class RaetschOuterCV(object):
    """Iterable over already separated CV partitions of the dataset."""

    def __init__(
        self,
        X: np.typing.NDArray[float],
        y: np.typing.NDArray[Union[int, float]],
        train_splits: Sequence[np.typing.NDArray[int]],
        test_splits: Sequence[np.typing.NDArray[int]],
    ) -> None:
        self.X = X
        self.y = y
        self.train_splits = train_splits
        self.test_splits = test_splits

    def __iter__(self) -> Iterator[Tuple[
        np.typing.NDArray[float],
        np.typing.NDArray[Union[int, float]],
        np.typing.NDArray[float],
        np.typing.NDArray[Union[int, float]],
    ]]:
        return (
            (self.X[tr - 1], self.y[tr - 1], self.X[ts - 1], self.y[ts - 1])
            for tr, ts in zip(self.train_splits, self.test_splits)
        )


def _fetch_remote(data_home: Optional[str] = None) -> Path:
    """
    Helper function to download the remote dataset into path.

    Fetch the remote dataset, save into path using remote's filename and ensure
    its integrity based on the SHA256 Checksum of the downloaded file.

    Parameters
    ----------
    dirname : string
        Directory to save the file to.

    Returns
    -------
    file_path: string
        Full path of the created file.
    """
    file_path = fetch_file(
        'raetsch',
        'https://github.com/tdiethe/gunnar_raetsch_benchmark_datasets'
        '/raw/master/benchmarks.mat',
        data_home=data_home,
    )
    sha256hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            buffer = f.read(8192)
            if not buffer:
                break
            sha256hash.update(buffer)
    checksum = sha256hash.hexdigest()
    remote_checksum = (
        '47c19e4bc4716edc4077cfa5ea61edf4d02af4ec51a0ecfe035626ae8b561c75'
    )
    if remote_checksum != checksum:
        raise IOError(
            f"{file_path} has an SHA256 checksum ({checksum}) differing "
            f"from expected ({remote_checksum}), file may be corrupted.",
        )
    return file_path


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
) -> Tuple[np.typing.NDArray[float], np.typing.NDArray[Union[int, float]]]:
    pass


def fetch(
    name: str,
    data_home: Optional[str] = None,
    *,
    return_X_y: bool = False,
) -> Union[
    Bunch,
    Tuple[np.typing.NDArray[float], np.typing.NDArray[Union[int, float]]],
]:
    """Fetch Gunnar Raetsch's dataset.

    Fetch a Gunnar Raetsch's benchmark dataset by name. Availabe datasets are
    'banana', 'breast_cancer', 'diabetis', 'flare_solar', 'german', 'heart',
    'image', 'ringnorm', 'splice', 'thyroid', 'titanic', 'twonorm' and
    'waveform'. More info at
    https://github.com/tdiethe/gunnar_raetsch_benchmark_datasets.

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
    if name not in DATASETS:
        raise Exception('Avaliable datasets are ' + str(list(DATASETS)))
    filename = _fetch_remote(data_home=data_home)
    X, y, train_splits, test_splits = loadmat(filename)[name][0][0]
    if len(y.shape) == 2 and y.shape[1] == 1:
        y = y.ravel()

    cv = RaetschOuterCV(X, y, train_splits, test_splits)

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
        DESCR=name,
    )
