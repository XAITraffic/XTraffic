"""
Test the Forex loader.

@author: David Diaz Vico
@license: MIT
"""

from datetime import date

from skdatasets.repositories.forex import fetch


def test_forex_usd_eur():
    """Tests forex USD-EUR dataset."""
    data = fetch(start=date(2015, 1, 1), end=date(2015, 1, 31),
                 currency_1='USD', currency_2='EUR')
    assert data.data.shape == (31, 1)


def test_forex_usd_eur_return_X_y():
    """Tests forex USD-EUR dataset."""
    X, y = fetch(start=date(2015, 1, 1), end=date(2015, 1, 31),
                 currency_1='USD', currency_2='EUR', return_X_y=True)
    assert X.shape == (31, 1)
    assert y is None


def test_forex_btc_eur():
    """Tests forex BTC-EUR dataset."""
    data = fetch(start=date(2015, 1, 1), end=date(2015, 1, 31),
                 currency_1='BTC', currency_2='EUR')
    assert data.data.shape == (31, 1)
