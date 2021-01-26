"""Module containing evaluation metrics."""
import scipy.stats
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder

from ..transformers.label import convert_array_to_closest_labels


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


def fix_shape(array):
    """Fixes the shape, if array is 2D but can be 1D.

    Args:
        array (np.array): input array.

    Returns (np.array): the fixed array.
    """
    if len(array.shape) == 2 and array.shape[1] == 1:
        return array.flatten()
    else:
        return array


def fix_shape_and_score(y_true, y_pred, scorer):
    """
    Fixes the shape of the input variables for correlation coefficients.

    Args:
        y_true: Ground truth (correct) target values.
        y_pred: Estimated targets as returned by a classifier.
        scorer: The scorer used.

    Returns: The result of the scorer.
    """
    return scorer(fix_shape(y_true), fix_shape(y_pred))


def extract_correlation(result):
    """Returns: the first entry in the tuple."""
    correlation, _ = result
    return correlation


def convert_labels(y_true, y_pred, scorer, converter):
    """Converts labels before applying the scorer.

    Args:
        y_true: Ground truth (correct) target values.
        y_pred: Estimated targets as returned by a classifier.
        scorer: The scorer used.
        converter: the converter used.

    Returns: the result of the scorer.
    """
    return scorer(converter(fix_shape(y_true)), converter(fix_shape(y_pred)))


def _convert_to_category(y_true, y_pred, categories, scorer):

    y_true = convert_array_to_closest_labels(fix_shape(y_true), categories)
    y_pred = convert_array_to_closest_labels(fix_shape(y_pred), categories)

    y_true = y_true.reshape(len(y_true), 1)
    y_pred = y_true.reshape(len(y_pred), 1)

    encoder = OneHotEncoder(categories=categories)

    y_true = encoder.fit_transform(y_true)
    y_pred = encoder.transform(y_pred)

    return scorer(y_true, y_pred)


def scoring(hit_nonhit_accuracy_score=hit_nonhit_accuracy_score,
            categories=None):
    """Returns a set of scoring functions used for evaluation.

    Args:
        hit_nonhit_accuracy_score: function returning hte hit non-hit score
        categories: list of labelscategoreis
    """
    scores = {
        'explained_variance':
            metrics.make_scorer(metrics.explained_variance_score),
        'neg_mean_absolute_error':
            metrics.make_scorer(
                metrics.mean_absolute_error,
                greater_is_better=False,
            ),
        'neg_mean_squared_error':
            metrics.make_scorer(
                metrics.mean_squared_error,
                greater_is_better=False,
            ),
        'neg_median_absolute_error':
            metrics.make_scorer(
                metrics.median_absolute_error,
                greater_is_better=False,
            ),
        'r2':
            metrics.make_scorer(metrics.r2_score),
        'pearsonr':
            metrics.make_scorer(
                lambda y_true, y_pred: extract_correlation(
                    fix_shape_and_score(
                        y_true,
                        y_pred,
                        scipy.stats.pearsonr,
                    )),
            ),
        'spearmanr':
            metrics.make_scorer(
                lambda y_true, y_pred: extract_correlation(
                    fix_shape_and_score(
                        y_true,
                        y_pred,
                        scipy.stats.spearmanr,
                    )),
            ),
        'kendalltau':
            metrics.make_scorer(
                lambda y_true, y_pred: extract_correlation(
                    fix_shape_and_score(
                        y_true,
                        y_pred,
                        scipy.stats.kendalltau,
                    )),
            ),
    }

    if hit_nonhit_accuracy_score is not None:
        scores['hit_nonhit_accuracy'] = hit_nonhit_accuracy_score

    if categories is not None:

        def label_converter(arr):
            return convert_array_to_closest_labels(arr, categories)

        scores['neg_mean_absolute_error_labels'] = metrics.make_scorer(
            lambda y_true, y_pred: convert_labels(
                y_true,
                y_pred,
                metrics.mean_absolute_error,
                label_converter,
            ),
            greater_is_better=False,
        )
        scores['neg_mean_squared_error_labels'] = metrics.make_scorer(
            lambda y_true, y_pred: convert_labels(
                y_true,
                y_pred,
                metrics.mean_squared_error,
                label_converter,
            ),
            greater_is_better=False,
        )
        scores['pearsonr_labels'] = metrics.make_scorer(
            lambda y_true, y_pred: extract_correlation(
                convert_labels(
                    y_true,
                    y_pred,
                    scipy.stats.pearsonr,
                    label_converter,
                )),
        )
        scores['spearmanr_labels'] = metrics.make_scorer(
            lambda y_true, y_pred: extract_correlation(
                convert_labels(
                    y_true,
                    y_pred,
                    scipy.stats.spearmanr,
                    label_converter,
                )),
        )
        scores['kendalltau_labels'] = metrics.make_scorer(
            lambda y_true, y_pred: extract_correlation(
                convert_labels(
                    y_true,
                    y_pred,
                    scipy.stats.kendalltau,
                    label_converter,
                )),
        )

        scores['confusion_matrix'] = metrics.make_scorer(
            lambda y_true, y_pred: _convert_to_category(
                y_true,
                y_pred,
                categories,
                metrics.confusion_matrix,
            ),
        )

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
