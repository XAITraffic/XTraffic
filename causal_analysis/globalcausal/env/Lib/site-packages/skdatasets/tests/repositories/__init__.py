"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def check_estimator(data):
    """Check that the dataset can be used to cross-validate an estimator."""
    estimator = GridSearchCV(Pipeline([('tr', StandardScaler(with_mean=False)),
                                       ('pred', Ridge(max_iter=4))]),
                             {'pred__alpha': [0.33, 0.66]},
                             cv=data.inner_cv, error_score=np.nan)
    if data.train_indices and data.test_indices:

        train_indices = data.train_indices

        train_indices += data.validation_indices

        estimator.fit(
            data.data[train_indices],
            y=data.target[train_indices],
        )
        estimator.score(
            data.data[data.test_indices],
            y=data.target[data.test_indices]
        )
    else:
        if hasattr(data.outer_cv, '__iter__'):
            for X, y, X_test, y_test in data.outer_cv:
                estimator.fit(X, y=y)
                estimator.score(X_test, y=y_test)
        else:
            cross_validate(estimator, data.data, y=data.target,
                           cv=data.outer_cv)
