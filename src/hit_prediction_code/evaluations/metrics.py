# -*- coding: utf-8 -*-
"""Module containing evaluation metrics."""
from sklearn import metrics


def hit_nonhit_accuracy_score(estimator,
                              x,
                              y,
                              normalize=True,
                              threshold=100.5):
    """Metric calculating hit/non-hit accuracy based on the regression value.

    Args:
        estimator: the estimator providing the predictions.
        x: the test samples.
        y: the expected target values.
        normalize: same as normalize of sklearns accuracy_score.
        threshold: the decission boundary between hits and non-hits.

    Returns:
        The computed accuracy score using sklearns accuracy_score after
        converting the regression values to hit/non-hit labels.

    """
    y_true = y > threshold
    y_pred = estimator.predict(x) > threshold

    return metrics.accuracy_score(y_true, y_pred, normalize)


def scoring():
    """Returns a set of scoring functions used for evaluation."""
    return {
        'explained_variance':
            metrics.make_scorer(metrics.explained_variance_score),
        'neg_mean_absolute_error':
            metrics.make_scorer(metrics.mean_absolute_error,
                                greater_is_better=False),
        'neg_mean_squared_error':
            metrics.make_scorer(metrics.mean_squared_error,
                                greater_is_better=False),
        'neg_median_absolute_error':
            metrics.make_scorer(metrics.median_absolute_error,
                                greater_is_better=False),
        'r2':
            metrics.make_scorer(metrics.r2_score),
        'hit_nonhit_accuracy':
            hit_nonhit_accuracy_score,
    }
