"""Tests for the z_order_index."""
import numpy as np

from hit_prediction_code.transformers.mel_spect import FloatListToSentence


def test_shapes():
    """Tests if data has a correct shape."""
    data = np.arange(0., 10., 1 / 3).reshape((3, 10))
    data = FloatListToSentence().fit_transform(data)

    assert data.shape == (3,)


def test_order():
    """Tests if the order is correct."""
    data = np.arange(0., 5., 1 / 8).reshape((4, 10))
    data = FloatListToSentence().fit_transform(data)

    expected = [
        '0.0 0.125 0.25 0.375 0.5 0.625 0.75 0.875 1.0 1.125',
        '1.25 1.375 1.5 1.625 1.75 1.875 2.0 2.125 2.25 2.37',
        '2.5 2.625 2.75 2.875 3.0 3.125 3.25 3.375 3.5 3.625',
        '3.75 3.875 4.0 4.125 4.25 4.375 4.5 4.625 4.75 4.87',
    ]

    assert (data == expected).all()


def test_round():
    """Tests if round works."""
    data = np.arange(0., 5., 1 / 8).reshape((4, 10))
    data = FloatListToSentence(round_decimals=2).fit_transform(data)

    expected = [
        '0.0 0.12 0.25 0.38 0.5 0.62 0.75 0.88 1.0 1.12',
        '1.25 1.38 1.5 1.62 1.75 1.88 2.0 2.12 2.25 2.3',
        '2.5 2.62 2.75 2.88 3.0 3.12 3.25 3.38 3.5 3.62',
        '3.75 3.88 4.0 4.12 4.25 4.38 4.5 4.62 4.75 4.8',
    ]

    print(data)

    assert (data == expected).all()
