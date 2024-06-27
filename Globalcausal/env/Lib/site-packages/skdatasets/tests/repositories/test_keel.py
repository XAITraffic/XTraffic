"""
Test the Keel loader.

@author: David Diaz Vico
@license: MIT
"""

from skdatasets.repositories.keel import fetch

from . import check_estimator


def check(data, shape, splits=1):
    """Check dataset properties."""
    assert data.data.shape == shape
    assert data.target.shape[0] == shape[0]
    if splits > 1:
        assert len(list(data.outer_cv)) == splits
    else:
        assert data.outer_cv is None
    assert not data.train_indices
    assert not data.validation_indices
    assert not data.test_indices
    assert data.inner_cv is None
    check_estimator(data)


def test_fetch_keel_abalone9_18():
    """Tests Keel abalone9-18 dataset."""
    data = fetch('imbalanced', 'abalone9-18')
    check(data, (731, 10))


def test_fetch_keel_abalone9_18_return_X_y():
    """Tests Keel abalone9-18 dataset."""
    X, y = fetch('imbalanced', 'abalone9-18', return_X_y=True)
    assert X.shape == (731, 10)
    assert y.shape == (731,)


def test_fetch_keel_abalone9_18_folds():
    """Tests Keel abalone9-18 dataset with folds."""
    data = fetch('imbalanced', 'abalone9-18', nfolds=5)
    check(data, (731, 10), splits=5)


def test_fetch_keel_banana():
    """Tests Keel banana dataset."""
    data = fetch('classification', 'banana')
    check(data, (5300, 2))


def test_fetch_keel_banana_folds():
    """Tests Keel banana dataset with folds."""
    data = fetch('classification', 'banana', nfolds=5)
    check(data, (5300, 2), splits=5)


def test_fetch_keel_banana_dobscv():
    """Tests Keel banana dataset with dobscv folds."""
    data = fetch('classification', 'banana', nfolds=5, dobscv=True)
    check(data, (5300, 2), splits=5)
