"""
@author: David Diaz Vico
@license: MIT
"""

import subprocess


def test_binary_classification():
    """Tests binary classification experiment."""
    ret = subprocess.call(['skdatasets/tests/utils/run.py', '-r', 'keel', '-c',
                           'imbalanced', '-d', 'abalone9-18', '-e',
                           'skdatasets/tests/utils/MLPClassifier.json'])
    assert ret >= 0
    ret = subprocess.call(['skdatasets/tests/utils/run.py', '-r', 'libsvm', '-c', 'binary',
                           '-d', 'breast-cancer', '-e',
                           'skdatasets/tests/utils/MLPClassifier.json'])
    assert ret >= 0
    ret = subprocess.call(['skdatasets/tests/utils/run.py', '-r', 'raetsch', '-d',
                           'banana', '-e', 'skdatasets/tests/utils/MLPClassifier.json'])
    assert ret >= 0


def test_multiclass_classification():
    """Tests multiclass classification experiment."""
    ret = subprocess.call(['skdatasets/tests/utils/run.py', '-r', 'sklearn', '-d', 'iris',
                           '-e', 'skdatasets/tests/utils/MLPClassifier.json'])
    assert ret >= 0
    ret = subprocess.call(['skdatasets/tests/utils/run.py', '-r', 'uci', '-d', 'wine',
                           '-e', 'skdatasets/tests/utils/MLPClassifier.json'])
    assert ret >= 0
    ret = subprocess.call(['skdatasets/tests/utils/run.py', '-r', 'libsvm', '-c',
                           'multiclass', '-d', 'shuttle', '-e',
                           'skdatasets/tests/utils/MLPClassifier.json'])
    assert ret >= 0
    ret = subprocess.call(['skdatasets/tests/utils/run.py', '-r', 'libsvm', '-c',
                           'multiclass', '-d', 'usps', '-e',
                           'skdatasets/tests/utils/MLPClassifier.json'])
    assert ret >= 0


def test_regression():
    """Tests regression experiment."""
    ret = subprocess.call(['skdatasets/tests/utils/run.py', '-r', 'libsvm', '-c',
                           'regression', '-d', 'housing', '-e',
                           'skdatasets/tests/utils/MLPRegressor.json'])
    assert ret >= 0
