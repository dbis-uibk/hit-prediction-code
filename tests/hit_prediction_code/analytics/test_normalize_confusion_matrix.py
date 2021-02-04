"""Tests the normalize confusion matrix function."""
import numpy as np
import pytest

from hit_prediction_code.analytics import normalize_confusion_matrix


def test_square_input_shape_check():
    """Tests if the input shape is checked to be a square."""
    with pytest.raises(AssertionError):
        normalize_confusion_matrix(cm=np.zeros((2, 1)), method='all')


def test_method_check():
    """Tests if an exceptions is raised when an unknown method is used."""
    with pytest.raises(ValueError):
        normalize_confusion_matrix(cm=np.zeros((3, 3)), method='unknown')


def test_true_normalization():
    """Tests if true normalization works."""
    cm = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    expected_outcome = np.array([
        [1 / 6, 2 / 6, 3 / 6],
        [4 / 15, 5 / 15, 6 / 15],
        [7 / 24, 8 / 24, 9 / 24],
    ])

    actual_outcome = normalize_confusion_matrix(cm, method='true')

    np.testing.assert_equal(actual_outcome, expected_outcome)


def test_pred_normalization():
    """Tests if pred normalization works."""
    cm = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    expected_outcome = np.array([
        [1 / 12, 2 / 15, 3 / 18],
        [4 / 12, 5 / 15, 6 / 18],
        [7 / 12, 8 / 15, 9 / 18],
    ])

    actual_outcome = normalize_confusion_matrix(cm, method='pred')

    np.testing.assert_equal(actual_outcome, expected_outcome)


def test_all_normalization():
    """Tests if all normalization works."""
    cm = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    expected_outcome = np.array([
        [1 / 45, 2 / 45, 3 / 45],
        [4 / 45, 5 / 45, 6 / 45],
        [7 / 45, 8 / 45, 9 / 45],
    ])

    actual_outcome = normalize_confusion_matrix(cm, method='all')

    np.testing.assert_equal(actual_outcome, expected_outcome)
