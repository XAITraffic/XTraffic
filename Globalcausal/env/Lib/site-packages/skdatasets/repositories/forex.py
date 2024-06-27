"""
Forex datasets (http://forex-python.readthedocs.io).

@author: David Diaz Vico
@license: MIT
"""

import time
from datetime import date, timedelta

import numpy as np
from sklearn.utils import Bunch

from forex_python.bitcoin import BtcConverter
from forex_python.converter import CurrencyRates


def _fetch(get_rate, start=date(2015, 1, 1), end=date.today()):
    """Fetch dataset."""
    data = []
    delta = end - start
    for d in range(delta.days + 1):
        day = start + timedelta(days=d)
        rate = get_rate(day)
        data.append(rate)
    return np.asarray(data).reshape((-1, 1))


def _load_bitcoin(start=date(2015, 1, 1), end=date.today(), currency='EUR'):
    """Load bitcoin dataset"""
    btcc = BtcConverter()

    def get_rate(day):
        return btcc.get_previous_price(currency, day)

    return _fetch(get_rate, start=start, end=end)


def _load_forex(start=date(2015, 1, 1), end=date.today(), currency_1='USD',
                currency_2='EUR'):
    """Load forex dataset."""
    cr = CurrencyRates()

    def get_rate(day):
        time.sleep(0.1)
        return cr.get_rate(currency_1, currency_2, day)

    return _fetch(get_rate, start=start, end=end)


def fetch(start=date(2015, 1, 1), end=date.today(), currency_1='USD',
          currency_2='EUR', return_X_y=False):
    """Fetch Forex datasets.

    Fetches the ECB Forex and Coindesk Bitcoin datasets. More info at
    http://forex-python.readthedocs.io.

    Parameters
    ----------
    start : date, default=2015-01-01
        Initial date.
    end : date, default=today
        Final date.
    currency_1 : str, default='USD'
        Currency 1.
    currency_2 : str, default='EUR'
        Currency 2.
    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object.

    Returns
    -------
    data : Bunch
        Dictionary-like object with all the data and metadata.

    (data, target) : tuple if ``return_X_y`` is True

    """
    if currency_1 == 'BTC':
        X = _load_bitcoin(start=start, end=end, currency=currency_2)
        descr = 'BTC-' + str(currency_2)
    elif currency_2 == 'BTC':
        X = _load_bitcoin(start=start, end=end, currency=currency_1)
        descr = 'BTC-' + str(currency_1)
    else:
        X = _load_forex(start=start, end=end, currency_1=currency_1,
                        currency_2=currency_2)
        descr = str(currency_1) + '-' + str(currency_2)
    descr = descr + start.strftime('%Y-%m-%d') + '-' + end.strftime('%Y-%m-%d')

    if return_X_y:
        return X, None

    return Bunch(
        data=X,
        target=None,
        train_indices=[],
        validation_indices=[],
        test_indices=[],
        inner_cv=None,
        outer_cv=None,
        DESCR=descr,
    )
