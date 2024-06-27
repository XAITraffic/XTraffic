"""
Test the Keras loader.

@author: David Diaz Vico
@license: MIT
"""

from skdatasets.repositories.keras import fetch

from . import check_estimator


def check(data, n_samples_train, n_samples_test, n_features):
    """Check dataset properties."""
    assert data.data.shape == (n_samples_train + n_samples_test, n_features)
    assert data.target.shape[0] == n_samples_train + n_samples_test
    assert len(data.train_indices) == n_samples_train
    assert len(data.test_indices) == n_samples_test
    assert not data.validation_indices


def test_keras_mnist():
    """Tests keras MNIST dataset."""
    data = fetch('mnist')
    check(data, n_samples_train=60000, n_samples_test=10000, n_features=28 * 28)


def test_keras_mnist_return_X_y():
    """Tests keras MNIST dataset."""
    X, y = fetch('mnist', return_X_y=True)
    assert X.shape == (70000, 28 * 28)
    assert y.shape == (70000,)
