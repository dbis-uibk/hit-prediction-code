"""Test for the convert_array_to_closest_labels converter."""
import numpy as np
import pytest

from hit_prediction_code.transformers.label import \
    convert_array_to_closest_labels


def test_array_shape_check():
    """Tests if an exception is raised for arrays with the wrong dim."""
    arr = np.random.rand(2, 2)

    with pytest.raises(ValueError):
        convert_array_to_closest_labels(arr, [1, 2])


def test_mapping_values_to_labels():
    """Tests if the values are mapped correctly."""
    arr = np.array([1.5, -1, 6, 2.25, 10, 15])
    labels = [0, 5, 10]

    expected_arr = np.array([0, 0, 5, 0, 10, 10])

    assert (convert_array_to_closest_labels(arr, labels) == expected_arr).all()
