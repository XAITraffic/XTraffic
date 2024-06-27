"""
Test the UCR loader.

@author: David Diaz Vico
@license: MIT
"""

from skdatasets.repositories.ucr import fetch


def test_fetch_ucr_gunpoint():
    """Tests UCR GunPoint dataset."""
    data = fetch('GunPoint')
    assert data.data.shape == (200, 150)
    assert len(data.train_indices) == 50
    assert len(data.test_indices) == 150


def test_fetch_ucr_gunpoint_return_X_y():
    """Tests UCR GunPoint dataset."""
    X, y = fetch('GunPoint', return_X_y=True)
    assert X.shape == (200, 150)
    assert y.shape == (200,)


def test_fetch_ucr_basicmotions():
    """Tests UCR GunPoint dataset."""
    data = fetch('BasicMotions')
    assert data.data.shape == (80,)
    assert len(data.train_indices) == 40
    assert len(data.test_indices) == 40
