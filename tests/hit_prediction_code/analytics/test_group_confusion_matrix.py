"""Tests the group_confusion matrix function."""
import numpy as np
import pytest

from hit_prediction_code.analytics import group_confusion_matrix


def test_input_dimension_check():
    """Tests if the input dimmension is checked to be a two."""
    with pytest.raises(AssertionError):
        group_confusion_matrix(np.zeros((2, 2, 1)), 1)


def test_square_input_shape_check():
    """Tests if the input shape is checked to be a square."""
    with pytest.raises(AssertionError):
        group_confusion_matrix(np.zeros((2, 1)), 1)


def test_group_size_check():
    """Tests if the check that all groups have the same size is in place."""
    with pytest.raises(ValueError):
        group_confusion_matrix(np.zeros((10, 10)), 3)


def test_grouping():
    """Tests if grouping works correctly."""
    matrix = np.arange(100).reshape(10, 10)
    expected_outcome = np.array([
        [22, 30, 38, 46, 54],
        [102, 110, 118, 126, 134],
        [182, 190, 198, 206, 214],
        [262, 270, 278, 286, 294],
        [342, 350, 358, 366, 374],
    ])

    actual_outcome = group_confusion_matrix(matrix, 5)

    assert (actual_outcome == expected_outcome).all()
