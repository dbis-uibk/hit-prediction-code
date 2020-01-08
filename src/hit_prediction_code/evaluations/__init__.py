from sklearn.model_selection import KFold

from . import metrics


def cv():
    return KFold(n_splits=5, shuffle=True, random_state=42)


def grid_parameters():
    return {
        'verbose': 3,
        'cv': cv(),
        'refit': False,
        'scoring': metrics.scoring(),
        'return_train_score': True,
    }
