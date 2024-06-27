"""
Tests.

@author: Carlos Ramos Carreño
@license: MIT
"""

from skdatasets.repositories.cran import fetch


def test_cran_geyser():
    """Tests CRAN geyser dataset."""
    fetch('geyser')


def test_cran_geyser_return_X_y():
    """Tests CRAN geyser dataset."""
    X, y = fetch('geyser', return_X_y=True)
