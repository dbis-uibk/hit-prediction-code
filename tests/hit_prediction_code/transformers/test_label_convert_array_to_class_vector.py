"""Tests for the convert_array_to_class_vector function."""
import numpy as np
import pytest

from hit_prediction_code.transformers.label import \
    convert_array_to_class_vector


def test_default_strategy():
    """Tests the default values."""
    labels = list(range(8))
    arr = np.array(2 * labels)

    expected_arr = []
    for row in range(8):
        expected_arr.append([1 if col <= row else 0 for col in range(8)])
    expected_arr = np.array(2 * expected_arr)

    actual_arr = convert_array_to_class_vector(arr, labels)

    assert (actual_arr == expected_arr).all()


def test_fill_strategy():
    """Tests the fill strategy."""
    strategy = 'fill'

    labels = list(range(0, 16, 2))
    arr = np.array(2 * labels)

    expected_arr = []
    for row in range(8):
        expected_arr.append([1 if col <= row else 0 for col in range(8)])
    expected_arr = np.array(2 * expected_arr)

    actual_arr = convert_array_to_class_vector(arr, labels, strategy=strategy)

    assert (actual_arr == expected_arr).all()


def test_one_hot_strategy():
    """Tests the one_hot strategy."""
    strategy = 'one_hot'

    labels = list(range(0, 16, 2))
    arr = np.array(2 * labels)

    expected_arr = []
    for row in range(8):
        expected_arr.append([1 if col == row else 0 for col in range(8)])
    expected_arr = np.array(2 * expected_arr)

    actual_arr = convert_array_to_class_vector(arr, labels, strategy=strategy)

    assert (actual_arr == expected_arr).all()


def test_array_shape_check():
    """Tests if array shape checks are in place."""
    arr = np.random.rand(2, 2)

    with pytest.raises(ValueError, match='shape'):
        convert_array_to_class_vector(arr, [1, 2])


def test_strategy_check():
    """Tests if strategy checks are in place."""
    unknown_strategy = 'wrong_strategy'

    labels = list(range(8))
    arr = np.array(2 * labels)

    with pytest.raises(ValueError, match=unknown_strategy):
        convert_array_to_class_vector(arr, labels, strategy=unknown_strategy)
