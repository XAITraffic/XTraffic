"""
@author: David Diaz Vico
@license: MIT
"""

from sklearn.model_selection import GridSearchCV

from skdatasets.utils.estimator import json2estimator


def test_json2estimator():
    """Tests instantiation of estimator from a json file."""
    import sklearn
    e = json2estimator('skdatasets/tests/utils/LinearRegression.json')
    assert type(e) == GridSearchCV


def test_json2estimator_custom():
    """Tests instantiation of a custom estimator from a json file."""
    import skdatasets
    e = json2estimator('skdatasets/tests/utils/LinearRegressionCustom.json')
    assert type(e) == GridSearchCV
