"""
Datasets from the UCR time series database.

@author: Carlos Ramos Carreño
@license: MIT
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Final, Literal, Optional, Sequence, Tuple, Union, overload

import numpy as np
import scipy.io.arff
from sklearn.utils import Bunch

from .base import fetch_zip as _fetch_zip

BASE_URL: Final = 'http://www.timeseriesclassification.com/Downloads/'


def _target_conversion(
    target: np.typing.NDArray[Union[int, str]],
) -> Tuple[np.typing.NDArray[int], Sequence[str]]:
    try:
        target_data = target.astype(int)
        target_names = np.unique(target_data).astype(str).tolist()
    except ValueError:
        target_names = np.unique(target).tolist()
        target_data = np.searchsorted(target_names, target)

    return target_data, target_names


def data_to_matrix(
    struct_array: np.typing.NDArray[object],
) -> np.typing.NDArray[float]:
    fields = struct_array.dtype.fields
    assert fields
    if(
        len(fields.items()) == 1
        and list(fields.items())[0][1][0] == np.dtype(np.object_)
    ):
        attribute = struct_array[list(fields.items())[0][0]]

        n_instances = len(attribute)
        n_curves = len(attribute[0])
        n_points = len(attribute[0][0])

        attribute_new = np.zeros(n_instances, dtype=np.object_)

        for i in range(n_instances):

            transformed_matrix = np.zeros((n_curves, n_points))

            for j in range(n_curves):
                for k in range(n_points):
                    transformed_matrix[j][k] = attribute[i][j][k]
                    attribute_new[i] = transformed_matrix

        return attribute_new

    else:
        return np.array(struct_array.tolist())


@overload
def fetch(
    name: str,
    *,
    data_home: Optional[str] = None,
    return_X_y: Literal[False] = False,
) -> Bunch:
    pass


@overload
def fetch(
    name: str,
    *,
    data_home: Optional[str] = None,
    return_X_y: Literal[True],
) -> Tuple[np.typing.NDArray[float], np.typing.NDArray[int]]:
    pass


def fetch(
    name: str,
    *,
    data_home: Optional[str] = None,
    return_X_y: bool = False,
) -> Union[
    Bunch,
    Tuple[np.typing.NDArray[float], np.typing.NDArray[int]],
]:
    """
    Fetch UCR dataset.

    Fetch a UCR dataset by name. More info at
    http://www.timeseriesclassification.com/.

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
    url = BASE_URL + name

    data_path = _fetch_zip(
        name,
        urlname=url + '.zip',
        subfolder="ucr",
        data_home=data_home,
    )

    description_filenames = [name, name + "Description", name + "_Info"]

    path_file_descr: Optional[Path]
    for f in description_filenames:
        path_file_descr = (data_path / f).with_suffix(".txt")
        if path_file_descr.exists():
            break
    else:
        # No description is found
        path_file_descr = None

    path_file_train = (data_path / (name + '_TRAIN')).with_suffix(".arff")
    path_file_test = (data_path / (name + '_TEST')).with_suffix(".arff")

    DESCR = (
        path_file_descr.read_text(errors='surrogateescape')
        if path_file_descr else ''
    )
    train = scipy.io.arff.loadarff(path_file_train)
    test = scipy.io.arff.loadarff(path_file_test)
    dataset_name = train[1].name
    column_names = np.array(train[1].names())
    target_column_name = column_names[-1]
    feature_names = column_names[column_names != target_column_name].tolist()
    target_column = train[0][target_column_name].astype(str)
    test_target_column = test[0][target_column_name].astype(str)
    y_train, target_names = _target_conversion(target_column)
    y_test, target_names_test = _target_conversion(test_target_column)
    assert target_names == target_names_test
    X_train = data_to_matrix(train[0][feature_names])
    X_test = data_to_matrix(test[0][feature_names])

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
        name=dataset_name,
        DESCR=DESCR,
        feature_names=feature_names,
        target_names=target_names,
    )
