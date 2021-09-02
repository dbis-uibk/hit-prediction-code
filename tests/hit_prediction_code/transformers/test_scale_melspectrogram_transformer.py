"""Tests for ScaleMelspectrogram transfomer."""
import numpy as np

from hit_prediction_code.transformers.mel_spect import MelSpectScaler


def test_default_scaling():
    """Tests the default parameters of the scaler."""
    data = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])

    actual = MelSpectScaler().fit_transform(data)
    expected = np.apply_along_axis(lambda x: x / x.max(), 0, data)

    assert np.all(actual >= 0.0)
    assert np.all(actual <= 1.0)
    np.testing.assert_allclose(actual, expected)


def test_min_value_scaling():
    """Tests the min value scaling of the scaler."""
    data = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])

    min_value = -1.0
    actual = MelSpectScaler(min_value=min_value).fit_transform(data)
    expected = np.apply_along_axis(
        lambda x: x * (1.0 - min_value) / x.max() + min_value, 0, data)

    assert np.all(actual >= min_value)
    assert np.all(actual <= 1.0)
    np.testing.assert_allclose(actual, expected)


def test_max_value_scaling():
    """Tests the max value scaling of the scaler."""
    data = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])

    max_value = 10.0
    actual = MelSpectScaler(max_value=max_value).fit_transform(data)
    expected = np.apply_along_axis(lambda x: x * max_value / x.max(), 0, data)

    assert np.all(actual >= 0.0)
    assert np.all(actual <= max_value)
    np.testing.assert_allclose(actual, expected)


def test_data_column():
    """Tests the column selection of the scaler."""
    data = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    data = np.array([data, data * 4])

    min_value = -1.0
    max_value = 10.0
    data_column = 1

    actual = MelSpectScaler(
        min_value=min_value,
        max_value=max_value,
        data_column=data_column,
    ).fit_transform(data)

    expected = data
    expected[data_column] = np.apply_along_axis(lambda x: x / x.max(), 0,
                                                expected[data_column])
    expected[data_column] *= (max_value - min_value)
    expected[data_column] += min_value

    assert np.all(actual >= 0.0)
    assert np.all(actual <= max_value)
    np.testing.assert_allclose(actual, expected)
