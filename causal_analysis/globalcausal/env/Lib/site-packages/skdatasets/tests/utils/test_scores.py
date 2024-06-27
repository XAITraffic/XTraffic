"""
@author: David Diaz Vico
@license: MIT
"""

import numpy as np

from skdatasets.utils.scores import hypotheses_table, scores_table

datasets = ['a4a', 'a8a', 'combined', 'dna', 'ijcnn1', 'letter', 'pendigits',
            'satimage', 'shuttle', 'usps', 'w7a', 'w8a']
estimators = ['LogisticRegression', 'MLPClassifier0', 'MLPClassifier1',
              'MLPClassifier2', 'MLPClassifier3', 'MLPClassifier4',
              'MLPClassifier5']
scores = np.asarray(((89.79, 89.78, 89.76, 89.88, 89.85, 89.91, 89.93),
                     (90.73, 90.73, 90.73, 90.85, 90.83, 90.81, 90.80),
                     (92.36, 92.31, 94.58, 94.82, 94.84, 94.92, 94.89),
                     (99.28, 99.27, 99.28, 99.26, 99.27, 99.25, 99.25),
                     (91.34, 91.34, 99.29, 99.33, 99.34, 99.53, 99.54),
                     (98.07, 98.04, 99.94, 99.95, 99.96, 99.96, 99.95),
                     (99.17, 99.08, 99.87, 99.87, 99.88, 99.90, 99.89),
                     (96.67, 96.28, 98.84, 98.87, 98.90, 98.87, 98.92),
                     (95.85, 92.83, 99.88, 99.93, 99.96, 99.98, 99.99),
                     (99.12, 99.11, 99.65, 99.58, 99.58, 99.65, 99.60),
                     (95.93, 95.40, 94.58, 96.31, 96.34, 96.58, 96.50),
                     (95.80, 95.99, 95.35, 96.20, 96.22, 96.36, 96.71)))


def test_scores_table() -> None:
    """Tests scores table."""
    scores_table(scores, datasets=datasets, estimators=estimators)
    scores_table(
        scores,
        stds=scores / 10.0,
        datasets=datasets,
        estimators=estimators,
    )


def test_hypotheses_table() -> None:
    """Tests hypotheses table."""
    for multitest in ('kruskal', 'friedmanchisquare', None):
        for test in ('mannwhitneyu', 'wilcoxon'):
            hypotheses_table(scores, estimators,
                             multitest=multitest, test=test)
            for correction in ('bonferroni', 'sidak', 'holm-sidak', 'holm',
                               'simes-hochberg', 'hommel', 'fdr_bh', 'fdr_by',
                               'fdr_tsbh', 'fdr_tsbky'):
                hypotheses_table(scores, estimators, multitest=multitest,
                                 test=test, correction=correction)
