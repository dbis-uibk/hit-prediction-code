"""Module containing random dataloaders."""
from dbispipeline.base import Loader
import numpy as np


class RandomMelSpectLoader(Loader):
    """Loads dataset with hits and non-hits contaning random data."""

    def __init__(self,
                 num_samples,
                 mel_shape,
                 hit_range=(1, 100),
                 non_hit_value=150):
        """Initializes the mel spect loader.

        Args:
            num_samples: the number of generated sample.
            mel_shape: the shape of the mel spectrogram.
            hit_range: the range to draw hit labels from.
            non_hit_value: the values used for NaN values in the dataset.
        """
        if num_samples % 2 != 0:
            raise ValueError('num_samples needs to be even.')

        self._data_shape = (num_samples, *mel_shape)
        self._hit_range = hit_range
        self._non_hit_value = non_hit_value
        self._config = {
            'num_samples': num_samples,
            'mel_shape': mel_shape,
            'data_shape': self._data_shape,
            'hit_range': hit_range,
            'nan_value': non_hit_value,
        }

    def load(self):
        """Returns the data loaded by the dataloader."""
        num_hit_labels = int(self._data_shape[0] * .5)
        num_non_hit_labels = num_hit_labels

        labels = np.concatenate(
            (
                np.random.random_integers(
                    low=self._hit_range[0],
                    high=self._hit_range[1],
                    size=(num_hit_labels,),
                ),
                np.full(
                    shape=(num_non_hit_labels,),
                    fill_value=self._non_hit_value,
                ),
            ),
            axis=0,
        )
        np.random.shuffle(labels)

        return np.random.random(size=self._data_shape), labels

    @property
    def configuration(self):
        """Returns the configuration in json serializable format."""
        return self._config


class RandomFeatureLoader(Loader):
    """Replaces the loades features with random data."""

    def __init__(self, dataloader):
        """Uses a dataloader to get data.

        Args:
            dataloader: used dataloader.
        """
        self._config = {
            'dataloader': dataloader.__class__.__name__,
            'dataloader_config': dataloader.configuration,
        }
