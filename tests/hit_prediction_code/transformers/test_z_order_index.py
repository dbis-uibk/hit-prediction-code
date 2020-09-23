"""Tests for the z_order_index."""
from hit_prediction_code.transformers.mel_spect import z_order_index


def test_shapes():
    """Tests if index has a correct shape."""
    assert len(z_order_index(rows=6, columns=4).shape) == 1


def test_order():
    """Tests if the order is correct."""
    order = z_order_index(rows=2, columns=2)
    assert (order == [0, 1, 2, 3]).all()

    order = z_order_index(rows=4, columns=2)
    assert (order == [0, 1, 2, 3, 4, 5, 6, 7]).all()

    order = z_order_index(rows=4, columns=4)
    assert (order == [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14,
                      15]).all()
