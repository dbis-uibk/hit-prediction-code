"""Tests for the ZOrderTransformer."""
import numpy as np

from hit_prediction_code.transformers.mel_spect import ZOrderTransformer


def _generate_2d_matrix(shape):
    matrix = np.arange(shape[0] * shape[1])
    return matrix.reshape(shape)


def test_shapes():
    """Tests if the reshape works correctly."""
    data = [
        _generate_2d_matrix(shape=(5, 4)),
        _generate_2d_matrix(shape=(5, 4)),
        _generate_2d_matrix(shape=(5, 4)),
    ]

    data = ZOrderTransformer().fit_transform(data)

    assert len(data.shape) == 2
    assert data.shape[0] == 3
    assert data.shape[1] == 20


def test_order():
    """Tests if the order is correct."""
    data = [
        _generate_2d_matrix(shape=(4, 4)),
        _generate_2d_matrix(shape=(4, 4)) + 3,
        _generate_2d_matrix(shape=(4, 4)),
    ]

    data = ZOrderTransformer().fit_transform(data)

    expected_order = [
        [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15],
        3 + np.array([0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15]),
        [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15],
    ]

    assert (data == expected_order).all()
