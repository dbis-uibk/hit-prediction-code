"""Tansformers for labels."""
from numbers import Real
from typing import List


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

    if value < previous_label:
        return previous_label

    for next_label in labels[1:]:
        if previous_label < value <= next_label:
            diff_to_previous = value - previous_label
            diff_to_next = next_label - value

            print(value, diff_to_previous, diff_to_next)
            if diff_to_previous < diff_to_next:
                return previous_label
            else:
                return next_label
        else:
            previous_label = next_label

    # return biggest label if there is no next label and value was
    # not between two labels
    return labels[-1]
