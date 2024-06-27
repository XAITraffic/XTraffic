"""
Scikit-learn datasets (http://scikit-learn.org/stable/datasets/index.html).

@author: David Diaz Vico
@license: MIT
"""

from sklearn.datasets import (fetch_20newsgroups,
                              fetch_20newsgroups_vectorized,
                              fetch_california_housing, fetch_covtype,
                              fetch_kddcup99, fetch_lfw_pairs,
                              fetch_lfw_people, fetch_olivetti_faces,
                              fetch_rcv1, load_breast_cancer, load_diabetes,
                              load_digits, load_iris, load_linnerud, load_wine,
                              make_biclusters, make_blobs, make_checkerboard,
                              make_circles, make_classification,
                              make_friedman1, make_friedman2, make_friedman3,
                              make_gaussian_quantiles, make_hastie_10_2,
                              make_low_rank_matrix, make_moons,
                              make_multilabel_classification, make_regression,
                              make_s_curve, make_sparse_coded_signal,
                              make_sparse_spd_matrix, make_sparse_uncorrelated,
                              make_spd_matrix, make_swiss_roll)

DATASETS = {'20newsgroups': fetch_20newsgroups,
            '20newsgroups_vectorized': fetch_20newsgroups_vectorized,
            'biclusters': make_biclusters, 'blobs': make_blobs,
            'breast_cancer': load_breast_cancer,
            'california_housing': fetch_california_housing,
            'checkerboard': make_checkerboard, 'circles': make_circles,
            'classification': make_classification, 'covtype': fetch_covtype,
            'diabetes': load_diabetes, 'digits': load_digits,
            'friedman1': make_friedman1, 'friedman2': make_friedman2,
            'friedman3': make_friedman3,
            'gaussian_quantiles': make_gaussian_quantiles,
            'hastie_10_2': make_hastie_10_2, 'iris': load_iris,
            'kddcup99': fetch_kddcup99, 'lfw_people': fetch_lfw_people,
            'lfw_pairs': fetch_lfw_pairs, 'linnerud': load_linnerud,
            'low_rank_matrix': make_low_rank_matrix, 'moons': make_moons,
            'multilabel_classification': make_multilabel_classification,
            'olivetti_faces': fetch_olivetti_faces, 'rcv1': fetch_rcv1,
            'regression': make_regression, 's_curve': make_s_curve,
            'sparse_coded_signal': make_sparse_coded_signal,
            'sparse_spd_matrix': make_sparse_spd_matrix,
            'sparse_uncorrelated': make_sparse_uncorrelated,
            'spd_matrix': make_spd_matrix, 'swiss_roll': make_swiss_roll,
            'wine': load_wine}


def fetch(name, *, return_X_y=False, **kwargs):
    """Fetch Scikit-learn dataset.

    Fetch a Scikit-learn dataset by name. More info at
    http://scikit-learn.org/stable/datasets/index.html.

    Parameters
    ----------
    name : string
        Dataset name.
    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object.
    **kwargs : dict
        Optional key-value arguments. See
        scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets.

    Returns
    -------
    data : Bunch
        Dictionary-like object with all the data and metadata.

    (data, target) : tuple if ``return_X_y`` is True

    """
    if return_X_y:
        kwargs["return_X_y"] = True

    data = DATASETS[name](**kwargs)

    if not return_X_y:
        data.train_indices = []
        data.validation_indices = []
        data.test_indices = []
        data.inner_cv = None
        data.outer_cv = None

    return data
