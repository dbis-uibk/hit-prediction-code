"""Transformers adapting the representation of mel_spectrograms."""
import numpy as np
import pymorton
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES


class MelSpectScaler(TransformerMixin, BaseEstimator):
    """Transformer scaling each mel-spectrogram."""

    def __init__(
        self,
        min_value=0.0,
        max_value=1.0,
        data_column=None,
    ):
        """Initializes the transformer.

        Args:
            min_value: the minimum value used for scaling.
            max_value: the maximum value used for scaling.
            data_column: index used to select a specific column if the data
                contains multiple features.
        """
        self.min_value = min_value
        self.max_value = max_value
        self.data_column = data_column

    def fit(self, data, target=None, **fit_params):
        """Does not do anything.

        Args:
            data: ignored.
            target: ignored.
            fit_params: ignored.

        Returns: Transformer instance.
        """
        return self

    def transform(self, data):
        """Transformers the data object.

        Args:
            data: data that is transformed.
        """
        data = check_array(
            data,
            copy=True,
            dtype=FLOAT_DTYPES,
            force_all_finite=True,
            ensure_2d=False,
            allow_nd=True,
        )

        if self.data_column is not None:
            data[self.data_column] = self._transform(data[self.data_column])
        else:
            data = self._transform(data)

        return data

    def _transform(self, data):
        return np.apply_along_axis(self._scale, 0, data)

    def _scale(self, data):
        data /= data.max()
        data *= (self.max_value - self.min_value)
        data += self.min_value
        return data


class ZOrderTransformer(TransformerMixin, BaseEstimator):
    """Reads the matrix and outputs the matrix serialized in z-order."""

    def __init__(self):
        """Creates the transformer."""

    def fit(self, data, target=None, **fit_params):
        """Does not do anything.

        Args:
            data: ignored.
            target: ignored.
            fit_params: ignored.

        Returns: Transformer instance.
        """
        return self

    def transform(self, data):
        """Transformers the data object.

        Args:
            data: data that is transformed.
        """
        data = check_array(
            data,
            copy=True,
            dtype=FLOAT_DTYPES,
            force_all_finite=True,
            ensure_2d=False,
            allow_nd=True,
        )

        shape = data.shape
        assert len(shape) == 3

        order = z_order_index_list_of_2d(data)
        data = data.ravel()
        data = data[order]

        return data.reshape((shape[0], shape[1] * shape[2]))


def z_order_index_list_of_2d(data):
    """Sorts each matrix in a list of 2d matrices by z-order."""
    assert len(data.shape) == 3

    order_2d = z_order_index(rows=data.shape[1], columns=data.shape[2])

    order_3d = []
    for row in range(data.shape[0]):
        offset = row * order_2d.shape[0]
        order_3d += list(offset + order_2d)

    return order_3d


def z_order_index(rows, columns):
    """Computes a z-order sorted array index.

    Args:
        rows: the number of rows,
        columns: the number ob columns.

    Returns: a 1d array containing indices that sort by z-order.
    """
    order = []
    for row in range(rows):
        for col in range(columns):
            order.append(pymorton.interleave2(col, row))

    return np.array(order).argsort()


class FloatListToSentence(TransformerMixin, BaseEstimator):
    """Converst a list of floats to Strings."""

    def __init__(self, round_decimals=None):
        """Creates the transformer object.

        Args:
            round_decimals: If specified, np.around is applied to the array
                and the given value is passed to the decimals parameter of
                np.around.
        """
        self.round_decimals = round_decimals

    def fit(self, data, target=None, **fit_params):
        """Does not do anything.

        Args:
            data: ignored.
            target: ignored.
            fit_params: ignored.

        Returns: Transformer instance.
        """
        return self

    def transform(self, data):
        """Transformers the data object.

        Floats are converted to str by using "%f".
        If specified round is applied as well.

        Args:
            data: data that is transformed.
        """
        data = check_array(
            data,
            copy=True,
            dtype=FLOAT_DTYPES,
            force_all_finite=True,
            ensure_2d=False,
            allow_nd=True,
        )

        assert len(data.shape) == 2

        if self.round_decimals:
            data = np.around(data, decimals=self.round_decimals)

        return np.apply_along_axis(self._array_to_sentence, 1, data)

    def _array_to_sentence(self, array):
        array = map(str, array)
        return ' '.join(array)
