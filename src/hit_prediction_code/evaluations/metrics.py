from sklearn import metrics


def hit_nonhit_accuracy_score(estimator, x, y, normalize=True,
                              threshold=100.5):
    y_true = y > threshold
    y_pred = estimator.predict(x) > threshold

    return metrics.accuracy_score(y_true, y_pred, normalize)


def scoring():
    return {
        'explained_variance':
            metrics.make_scorer(metrics.explained_variance_score),
        'neg_median_absolute_error':
            metrics.make_scorer(metrics.median_absolute_error,
                                greater_is_better=False),
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
