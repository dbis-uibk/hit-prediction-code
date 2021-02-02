"""Analytics code used to process results."""
import os.path
from typing import Dict, List

import pandas as pd


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
