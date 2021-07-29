"""Tansformers for labels."""
import numbers
from numbers import Real
from typing import List

import numpy as np
import pandas as pd


def convert_to_closest_label(value: Real, labels: List[int]) -> int:
    """
    Assigns a value to the closes value in a target list.

    Args:
        value (Real): to assign a label for.
        labels (List[int]): list of labels to use; needs to contain
            at least two labels.

    Returns:
        The closest label.
    """
    for elem in labels:
        assert type(elem) is int, 'at least one label is not type int'

    labels = sorted(labels)

    if len(labels) < 2:
        raise ValueError('Use at least two labels.')

    previous_label: int = labels[0]

    if value <= previous_label:
        return previous_label

    for next_label in labels[1:]:
        if previous_label < value <= next_label:
            diff_to_previous = value - previous_label
            diff_to_next = next_label - value

            if diff_to_previous < diff_to_next:
                return previous_label
            else:
                return next_label
        else:
            previous_label = next_label

    # return biggest label if there is no next label and value was
    # not between two labels
    return labels[-1]


def convert_array_to_closest_labels(array: np.array,
                                    labels: List[int]) -> np.array:
    """Converts the numpy array to labels.

    Args:
        array (np.array): the array to be converted.
        labels (List[int]): the list of target labels.
            vector.

    Returns (np.array): of the mapped labels.
    """
    if len(array.shape) != 1:
        raise ValueError('The array needs to be 1D shape.')

    return np.fromiter(
        map(lambda v: convert_to_closest_label(v, labels), array),
        dtype=np.int,
    )


def _convert_label_to_vector(label: numbers.Integral, strategy,
                             labels) -> List[int]:
    assert isinstance(label,
                      numbers.Integral), 'label argument has to be of type int'

    if strategy == 'fill':
        return [1 if elem <= label else 0 for elem in labels]
    elif strategy == 'one_hot':
        return [1 if elem == label else 0 for elem in labels]
    else:
        raise ValueError(f'Strategy {strategy} unknown.')


def convert_array_to_class_vector(array: np.array,
                                  labels: List[int],
                                  strategy='fill') -> np.array:
    """Converts the numpy array to a label vector.

    Args:
        array (np.array): the array to be converted.
        labels (List[int]): the list of target labels.
        strategy (str): the converting strategie used to create the class

    Returns:
        np.array: an array containing the class vectors.
    """
    array = convert_array_to_closest_labels(array=array, labels=labels)
    if len(array.shape) <= 1:
        array = array.reshape(len(array), 1)

    return np.apply_along_axis(
        lambda v: _convert_label_to_vector(
            v[0],
            strategy=strategy,
            labels=labels,
        ),
        1,
        array,
    )


def yang_hit_score(play_count: np.array, listener_count: np.array) -> np.array:
    """Computes the hit score as defined by Yang et al.

    Args:
        play_count (np.array): the play count.
        listener_count (np.array): the play count.

    Returns:
        np.array: the hit score.
    """
    return np.log(play_count) * np.log(listener_count)


def compute_hit_score_on_df(df: pd.DataFrame,
                            pc_column: str,
                            lc_column: str,
                            hit_score_column='yang_hit_score') -> pd.DataFrame:
    """Computes the hits core on a dataframe.

    Args:
        df (pd.DataFrame): dataframe that gets modified.
        pc_column (str): column name of the play count column.
        lc_column (str): column name of the listener count column.
        hit_score_column (str, optional): column name of the hit score.
    """
    df[hit_score_column] = yang_hit_score(df[[pc_column]], df[[lc_column]])
