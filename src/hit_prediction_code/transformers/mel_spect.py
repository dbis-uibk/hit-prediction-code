"""Trasformers adapting the representation of mel_spectrograms."""
import numpy as np
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
        """Trasformers the data object.

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
