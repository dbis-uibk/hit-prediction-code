"""Tansformers for labels."""
from numbers import Real
from typing import List

import numpy as np


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


def _convert_label_to_vector(label: int, strategy, label_count) -> List[int]:
    if strategy == 'fill':
        return [1 if elem <= label else 0 for elem in range(label_count)]
    elif strategy == 'one_hot':
        return [1 if elem == label else 0 for elem in range(label_count)]
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
            label_count=len(labels),
        ),
        1,
        array,
    )
