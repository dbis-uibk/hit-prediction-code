"""Tests for the confusion_matrix_to_multilabel_confusion_matrix function."""
import numpy as np
import pytest

from hit_prediction_code.analytics import \
    confusion_matrix_to_multilabel_confusion_matrix


def test_square_input_shape_check():
    """Tests if the input shape is checked to be a square."""
    with pytest.raises(AssertionError):
        confusion_matrix_to_multilabel_confusion_matrix(cm=np.zeros((2, 1)))


def test_resulting_shape():
    """Tests if the results has the correct shape."""
    cm = np.zeros((5, 5))
    expected_output_shape = (5, 2, 2)

    actual_output = confusion_matrix_to_multilabel_confusion_matrix(cm)

    assert actual_output.shape == expected_output_shape


def test_transformation():
    """Tests if the transformation is done right."""
    cm = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    expected_outcome = np.array([
        [
            [28, 11],
            [5, 1],
        ],
        [
            [20, 10],
            [10, 5],
        ],
        [
            [12, 9],
            [15, 9],
        ],
    ])

    actual_outcome = confusion_matrix_to_multilabel_confusion_matrix(cm)

    np.testing.assert_equal(actual_outcome, expected_outcome)
