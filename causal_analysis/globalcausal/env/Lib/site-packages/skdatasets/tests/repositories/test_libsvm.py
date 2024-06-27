"""
Test the LIBSVM loader.

@author: David Diaz Vico
@license: MIT
"""

from skdatasets.repositories.libsvm import fetch

from . import check_estimator


def check(
    data,
    n_features,
    n_samples=None,
    n_samples_train=None,
    n_samples_validation=None,
    n_samples_test=None,
    n_samples_remaining=None,
    estimator=True,
):
    """Check dataset properties."""
    if n_samples is None:
        n_samples = sum(n for n in [
            n_samples_train,
            n_samples_validation,
            n_samples_test,
            n_samples_remaining
        ] if n is not None)

    assert data.data.shape == (n_samples, n_features)
    assert data.target.shape[0] == n_samples

    if n_samples_train is None:
        assert not data.train_indices
    else:
        assert len(data.train_indices) == n_samples_train

    if n_samples_validation is None:
        assert not data.validation_indices
    else:
        assert len(data.validation_indices) == n_samples_validation

    if n_samples_test is None:
        assert not data.test_indices
    else:
        assert len(data.test_indices) == n_samples_test

    if n_samples_validation is None:
        assert data.inner_cv is None
    else:
        assert data.inner_cv is not None

    assert data.outer_cv is None

    if estimator:
        check_estimator(data)


def test_fetch_libsvm_australian():
    """Tests LIBSVM australian dataset."""
    data = fetch('binary', 'australian')
    check(data, n_samples=690, n_features=14)


def test_fetch_libsvm_australian_return_X_y():
    """Tests LIBSVM australian dataset."""
    X, y = fetch('binary', 'australian', return_X_y=True)
    assert X.shape == (690, 14)
    assert y.shape == (690,)


def test_fetch_libsvm_liver_disorders():
    """Tests LIBSVM liver-disorders dataset."""
    data = fetch('binary', 'liver-disorders')
    check(data, n_samples_train=145, n_samples_test=200, n_features=5)


def test_fetch_libsvm_duke():
    """Tests LIBSVM duke dataset."""
    data = fetch('binary', 'duke')
    check(data, n_samples_train=38, n_samples_validation=4,
          n_features=7129, estimator=False)


def test_fetch_libsvm_cod_rna():
    """Tests LIBSVM cod-rna dataset."""
    data = fetch('binary', 'cod-rna')
    check(data, n_samples_train=59535, n_samples_test=271617,
          n_samples_remaining=157413, n_features=8)


def test_fetch_libsvm_satimage():
    """Tests LIBSVM satimage dataset."""
    data = fetch('multiclass', 'satimage.scale')
    check(data, n_samples_train=3104, n_samples_test=2000,
          n_samples_validation=1331, n_features=36)
