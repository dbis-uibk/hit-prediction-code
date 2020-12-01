"""Module containing evaluation metrics."""
import scipy.stats
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


def hit_nonhit_score(estimator, x, y, scorer):
    """F1 score between hit and non-hit class."""
    # The classes are sorted as the LabelBinarizer does that. Hence, the last
    # label is the non-hit label and all others are hit labesl.
    return scorer(y[:, -1], estimator.predict(x)[:, -1])


def fix_shape(y_true, y_pred, scorer):
    """
    Fixes the shape of the input variables for correlation coefficients.

    Args:
        y_true: Ground truth (correct) target values.
        y_pred: Estimated targets as returned by a classifier.
        scorer: The scorer used.

    Returns: The result of the scorer.
    """
    if len(y_true.shape) == 2 and y_true.shape[1] == 1:
        y_true = y_true.flatten()

    if len(y_pred.shape) == 2 and y_pred.shape[1] == 1:
        y_pred = y_pred.flatten()

    return scorer(y_true, y_pred)


def scoring(hit_nonhit_accuracy_score=hit_nonhit_accuracy_score):
    """Returns a set of scoring functions used for evaluation."""
    scores = {
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
        'pearsonr':
            metrics.make_scorer(lambda y_true, y_pred: fix_shape(
                y_true, y_pred, scipy.stats.pearsonr)),
        'spearmanr':
            metrics.make_scorer(lambda y_true, y_pred: fix_shape(
                y_true, y_pred, scipy.stats.spearmanr)),
        'kendalltau':
            metrics.make_scorer(lambda y_true, y_pred: fix_shape(
                y_true, y_pred, scipy.stats.kendalltau)),
    }

    if hit_nonhit_accuracy_score is not None:
        scores['hit_nonhit_accuracy'] = hit_nonhit_accuracy_score

    return scores


def classifier_scoring():
    """Return a set of scoring functions for classification setup."""
    return {
        'hit_nonhit_accuracy':
            lambda e, x, y: hit_nonhit_score(
                estimator=e,
                x=x,
                y=y,
                scorer=metrics.accuracy_score,
            ),
        'hit_nonhit_f1':
            lambda e, x, y: hit_nonhit_score(
                estimator=e,
                x=x,
                y=y,
                scorer=metrics.f1_score,
            ),
        'hit_nonhit_recall':
            lambda e, x, y: hit_nonhit_score(
                estimator=e,
                x=x,
                y=y,
                scorer=metrics.recall_score,
            ),
        'hit_nonhit_precission':
            lambda e, x, y: hit_nonhit_score(
                estimator=e,
                x=x,
                y=y,
                scorer=metrics.precision_score,
            ),
        'f1_micro':
            metrics.make_scorer(metrics.f1_score, average='micro'),
        'f1_macro':
            metrics.make_scorer(metrics.f1_score, average='macro'),
        'recall_micro':
            metrics.make_scorer(metrics.recall_score, average='micro'),
        'recall_macro':
            metrics.make_scorer(metrics.recall_score, average='macro'),
        'precision_micro':
            metrics.make_scorer(metrics.precision_score, average='micro'),
        'precision_macro':
            metrics.make_scorer(metrics.precision_score, average='macro'),
        'multilabel_confusion_matrix':
            metrics.make_scorer(metrics.multilabel_confusion_matrix),
    }
