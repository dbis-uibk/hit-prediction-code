"""Tests for the z_order_index."""
import numpy as np

from hit_prediction_code.transformers.mel_spect import z_order_index_list_of_2d


def _generate_data(shape):
    size = 1
    for e in shape:
        size *= e

    return np.arange(size).reshape(shape)


def test_shapes():
    """Tests if index has a correct shape."""
    assert len(z_order_index_list_of_2d(_generate_data((3, 6, 4)))) == 72


def test_order():
    """Tests if the order is correct."""
    data = _generate_data(shape=(3, 2, 2))

    expected_order_2d = np.array([0, 1, 2, 3])
    expected_order = np.append(expected_order_2d, 4 + expected_order_2d)
    expected_order = np.append(expected_order, 8 + expected_order_2d)

    order = z_order_index_list_of_2d(data)

    assert (order == expected_order).all()

    data = _generate_data(shape=(3, 4, 2))

    expected_order_2d = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    expected_order = np.append(expected_order_2d, 8 + expected_order_2d)
    expected_order = np.append(expected_order, 16 + expected_order_2d)

    order = z_order_index_list_of_2d(data)

    assert (order == expected_order).all()

    data = _generate_data(shape=(3, 4, 4))

    expected_order_2d = np.array(
        [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15])
    expected_order = np.append(expected_order_2d, 16 + expected_order_2d)
    expected_order = np.append(expected_order, 32 + expected_order_2d)

    order = z_order_index_list_of_2d(data)

    assert (order == expected_order).all()
