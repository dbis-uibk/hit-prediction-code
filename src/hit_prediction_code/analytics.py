"""Analytics code used to process results."""
import os.path
from typing import Dict, List, Tuple

from matplotlib import pyplot
import numpy as np
import pandas as pd
import seaborn as sns


def _splits_from_outcome(outcome: pd.Series) -> List[str]:

    def check_number(v):
        try:
            int(v)
            return True
        except ValueError:
            return False

    if len(outcome) > 0:
        return [k for k in outcome.iloc[0].keys() if check_number(k)]
    else:
        return []


def aggregate_splits_per_epoch(outcome, aggregation_function):
    """
    Aggregates splits for each epoch.

    Args:
        outcome: the outcome as stored in the db.
        aggregation_function: the used aggregation function.

    Returns:
        A dataframe with aggregated results for all splits.
    """
    splits = _splits_from_outcome(outcome=outcome)

    result = []
    for row in outcome:
        data = {}
        for epoch in row[splits[0]].keys():
            data[epoch] = {}
            for metric in row[splits[0]][epoch].keys():
                values = []
                for split in splits:
                    values.append(row[split][epoch][metric])
                data[epoch][metric] = aggregation_function(values)
        result.append(data)

    return pd.DataFrame(result)


def aggregate_epochs(outcome, aggregation_function) -> pd.DataFrame:
    """
    Aggregates epochs.

    Args:
        outcome: the outcome in form of e.g. the result of a split aggregation.
        aggregation_function: the used aggregation function.

    Returns:
        A dataframe containing the result of the aggregation over the epochs.
    """
    metrics = outcome[outcome.columns[0]].iloc[0].keys()

    result = pd.DataFrame(columns=metrics)

    for metric in metrics:
        result[metric] = outcome.applymap(lambda v: v[metric]).apply(
            aggregation_function, axis=1)

    return result


def _extract_cv_epoch_evaluator_outcome_entry(
    outcome: pd.Series,
    key='mean',
) -> pd.Series:
    """Extracts an entry from the cv epoch evaluator results.

    The entry is either the mean, the standard deviation or a split.

    Args:
        results (pd.Series): the results dataframe containing the outcome.
        key (str, optional): the key name of the entry that should be extacted.
            Defaults to 'mean'.

    Returns:
        pd.Series: the extracted entry.
    """
    return outcome.apply(lambda v: pd.DataFrame(v[key]))


def _extract_splits_form_cv_epoch_evaluator_outcome(
        outcome: pd.Series) -> Dict[str, pd.Series]:
    splits = sorted(_splits_from_outcome(outcome=outcome))

    extracted = {}

    for split in splits:
        split_key = 'split-' + split
        extracted[split_key] = _extract_cv_epoch_evaluator_outcome_entry(
            outcome,
            key=split,
        )

    return extracted


def add_cv_epoch_evaluator_outcome_to_df(df: pd.DataFrame) -> None:
    """Adds all entries from cv epoch evaluator outcome as dataframe columns.

    Args:
        df (pd.DataFrame): the dataframe that is modified.
    """
    for key in ['mean', 'stdev']:
        df[key] = _extract_cv_epoch_evaluator_outcome_entry(
            outcome=df['outcome'],
            key=key,
        )

    for key, result in _extract_splits_form_cv_epoch_evaluator_outcome(
            outcome=df['outcome']).items():
        df[key] = result


def add_approach_to_df(df: pd.DataFrame) -> None:
    """Add an approach column to the dataframe.

    Args:
        df (pd.DataFrame): the dataframe that is modified.
    """
    df['approach'] = df['sourcefile'].apply(
        lambda v: os.path.splitext(os.path.basename(v))[0].replace(
            'unique_', '').replace('wide_and_deep', 'wd'))


def plot_epochs_confution_matrix(
        plot_title: str,
        data: List,
        columns: List,
        plot_shape=(2, 2),
        figsize=(16, 14),
) -> None:
    """Plots selected epoch confusion matrix as heatmaps.

    The epochs are selected by chunking the entries evenly.

    Args:
        plot_title (str): the title of the plot.
        data (List): the data used for ploting.
        columns (List): columns used to name the subplots.
        plot_shape (tuple, optional): the shape used to generate the subplots.
        figsize (tuple, optional): the figure size.
    """
    num_plots = plot_shape[0] * plot_shape[1]

    step = int(round(len(data) / num_plots, 0))
    idx = list(range(step, len(data), step))

    if len(idx) < num_plots:
        if len(idx) < num_plots - 1:
            idx.insert(0, 0)

        idx.append(len(data) - 1)
    else:
        idx = idx[:num_plots]
        idx[-1] = len(data) - 1

    assert num_plots == len(idx)

    fig = pyplot.figure(figsize=figsize)
    fig.suptitle(plot_title)

    max_value = float('-inf')
    for data_idx in idx:
        max_value = max(max_value, data[data_idx].max())

    for i, data_idx in enumerate(idx):
        plot = pyplot.subplot(*plot_shape, i + 1)
        sns.heatmap(data[data_idx], vmin=0, vmax=max_value)
        plot.set_title('%s Epoches' % columns[data_idx])
        plot.set_xlabel('predicted')
        plot.set_ylabel('actual')
        plot.xaxis.set_ticks_position('top')
        plot.xaxis.set_label_position('top')

    pyplot.show()


def group_confusion_matrix(matrix: np.array, num_classes: int) -> np.array:
    """Groups classes in confusion matrix.

    Args:
        matrix (np.array): the confusion matrix that should be grouped.
        num_classes (int): number of resulting classes.

    Raises:
        ValueError: if the number of classes present in the input matrix can
            not be evenly divided by the number of grouped classes.

    Returns:
        np.array: the array with grouped input classes.
    """
    assert len(matrix.shape) == 2
    assert matrix.shape[0] == matrix.shape[1]

    if (matrix.shape[0] % num_classes) != 0:
        raise ValueError('Confution matrix can not be grouped to %d classes' %
                         num_classes)

    for axis in [0, 1]:
        grouped_matrix = []

        for col_group in np.split(matrix, num_classes, axis=axis):
            grouped_matrix.append(np.sum(col_group, axis=axis))

        matrix = np.array(grouped_matrix)

    return np.transpose(matrix)


def normalize_confusion_matrix(cm: any, method: str) -> np.array:
    """Normalizes the confusion matrix.

    Args:
        cm (any): the confusion matrix that should be normalized
        method (str): the normalization method. Currently 'true', 'pred' and
            'all' is supported.

    Raises:
        ValueError: if an unknown mehtod is used.

    Returns:
        np.array: the normalized confusion matrix.
    """
    cm = np.array(cm)

    assert cm.shape[0] == cm.shape[1]

    with np.errstate(all='ignore'):
        if method == 'true':
            cm = cm / cm.sum(axis=1, keepdims=True)
        elif method == 'pred':
            cm = cm / cm.sum(axis=0, keepdims=True)
        elif method == 'all':
            cm = cm / cm.sum()
        else:
            raise ValueError('Method %s unknown' % method)
        cm = np.nan_to_num(cm)

    return cm


def confusion_matrix_to_multilabel_confusion_matrix(cm: any) -> np.array:
    """Transforms a confusion matrix into the multi-label version.

    Args:
        cm (any): confusion matrix to transform.

    Returns:
        np.array: the multi-label confusion matrix. Where index 0 in dimension
            0 represents the class 0. The 2x2 matrix for that class has the
            shape as the sklearn matrix. Meaning index 0 is the Negative class
            and index 1 the positive class.
    """
    cm = np.array(cm)

    assert cm.shape[0] == cm.shape[1]

    def _tn_fp_fn_tp_from_matrix(cm):
        tp = cm[0, 0]
        fn = np.sum(cm[0, 1:])
        fp = np.sum(cm[1:, 0])
        tn = np.sum(cm[1:, 1:])

        return tn, fp, fn, tp

    mcm = []
    for _ in range(cm.shape[0]):
        label_cm = np.array(_tn_fp_fn_tp_from_matrix(cm)).reshape(2, 2)
        mcm.append(label_cm)
        cm = np.roll(np.roll(cm, -1, axis=1), -1, axis=0)

    return np.array(mcm)


def _divide(numerator, denominator):
    denominator = denominator.copy()
    denominator[denominator == 0.] = 1  # avoid division by 0

    return numerator / denominator


def precision_recall_fscore(mcm: np.array,
                            average='macro') -> Tuple[float, float, float]:
    """Calculates precision, recall and f1 score.

    Args:
        mcm (np.array): the multilabel confusion matrix used to compute the
            scores.
        average (str, optional): the average method used. Currently 'micro'
            and 'macro' are supported.

    Raises:
        ValueError: if the average method is unknown.

    Returns:
        Tuple[float, float, float]: tuple containing the results for
            precision, recall and f1 score.
    """
    assert len(mcm.shape) == 3
    assert mcm.shape[1] == mcm.shape[2] == 2

    tp_sum = mcm[:, 1, 1]
    pred_sum = tp_sum + mcm[:, 0, 1]
    true_sum = tp_sum + mcm[:, 1, 0]

    if average == 'micro':
        tp_sum = np.array([tp_sum.sum()])
        pred_sum = np.array([pred_sum.sum()])
        true_sum = np.array([true_sum.sum()])

    precision = _divide(tp_sum, pred_sum)
    recall = _divide(tp_sum, true_sum)
    f_score = _divide(2 * precision * recall, precision + recall)

    if average == 'macro':
        precision = np.average(precision)
        recall = np.average(recall)
        f_score = np.average(f_score)
    elif not average == 'micro':
        raise ValueError('Unknown average %s' % average)

    return precision, recall, f_score
