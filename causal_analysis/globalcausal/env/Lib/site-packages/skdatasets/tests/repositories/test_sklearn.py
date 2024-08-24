"""
Test the Scikit-learn loader.

@author: David Diaz Vico
@license: MIT
"""

from skdatasets.repositories.sklearn import fetch

from . import check_estimator


def test_sklearn_iris():
    """Tests Scikit-learn iris dataset."""
    data = fetch('iris')
    assert data.data.shape == (150, 4)
    check_estimator(data)


def test_sklearn_iris_return_X_y():
    """Tests Scikit-learn iris dataset."""
    X, y = fetch('iris', return_X_y=True)
    assert X.shape == (150, 4)
    assert y.shape == (150,)
