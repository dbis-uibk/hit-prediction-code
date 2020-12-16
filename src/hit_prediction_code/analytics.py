"""Analytics code used to process results."""
import pandas as pd


def aggregate_splits_per_epoch(outcome, aggregation_function):
    """
    Aggregates splits for each epoch.

    Args:
        outcome: the outcome as stored in the db.
        aggregation_function: the used aggregation function.

    Returns:
        A dataframe with aggregated results for all splits.
    """

    def check_number(v):
        try:
            int(v)
            return True
        except ValueError:
            return False

    splits = [k for k in outcome.iloc[0].keys() if check_number(k)]

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


def aggregate_epochs(outcome, aggregation_function):
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
