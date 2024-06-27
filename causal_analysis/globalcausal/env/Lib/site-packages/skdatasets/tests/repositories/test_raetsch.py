"""
Test the Raetsch loader.

@author: David Diaz Vico
@license: MIT
"""

from skdatasets.repositories.raetsch import fetch

from . import check_estimator


def check(data, shape, splits=100):
    """Check dataset properties."""
    assert data.data.shape == shape
    assert data.target.shape[0] == shape[0]
    assert len(list(data.outer_cv)) == splits
    check_estimator(data)


def test_fetch_raetsch_banana():
    """Tests Gunnar Raetsch banana dataset."""
    data = fetch('banana')
    check(data, (5300, 2), splits=100)


def test_fetch_raetsch_banana_return_X_y():
    """Tests Gunnar Raetsch banana dataset."""
    X, y = fetch('banana', return_X_y=True)
    assert X.shape == (5300, 2)
    assert y.shape == (5300,)
