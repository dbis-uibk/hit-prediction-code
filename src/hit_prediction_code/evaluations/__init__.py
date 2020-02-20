# -*- coding: utf-8 -*-
"""Evaluation packages providing evaluation utils."""
from sklearn.model_selection import KFold

from . import metrics


def cv():
    """Returns a commonly used cross validation setup for splitting."""
    return KFold(n_splits=5, shuffle=True, random_state=42)


def grid_parameters():
    """Returns commonly used gird parameters used for gird search."""
    return {
        'verbose': 3,
        'cv': cv(),
        'refit': False,
        'scoring': metrics.scoring(),
        'return_train_score': True,
    }
