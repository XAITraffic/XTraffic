"""
@author: David Diaz Vico
@license: MIT
"""

import jsonpickle


def json2estimator(estimator, **kwargs):
    """Instantiate a Scikit-learn estimator from a json file.

    Instantiate a Scikit-learn estimator from a json file passing its path as
    argument.

    Parameters
    ----------
    estimator : str
        Path of the json file containing the estimator specification.
    **kwargs : dict
        Dictionary of optional keyword arguments.

    Returns
    -------
    estimator : Estimator
        Instantiated Scikit-learn estimator.

    """
    with open(estimator, 'r') as definition:
        estimator = jsonpickle.decode(definition.read())
        for k, v in kwargs.items():
            setattr(estimator, k, v)
        return estimator
