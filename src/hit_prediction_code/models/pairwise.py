"""Pairwise model wrappers."""
import random

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

from hit_prediction_code.transformers.label import \
    convert_array_to_class_vector
from hit_prediction_code.transformers.pairwise import PairwiseTransformer


class PairwiseOrdinalModel(ClassifierMixin, BaseEstimator):
    """This class wraps a model to predict classes on pairs of samples."""

    def __init__(self,
                 wrapped_model: BaseEstimator,
                 epochs: int = 1,
                 pairs_factor: float = 1.,
                 threshold_type: str = 'random',
                 pair_strategy: str = 'random',
                 pair_encoding: str = 'concat',
                 threshold_sample_training=False) -> None:
        """Creates the wrapper.

        Args:
            wrapped_model (BaseEstimator): the model used for the actual
                prediction.
            epochs (int): number of epochs to train.
            pairs_factor (float): factor multiplied with the length of data.
            threshold_type (str): how to compute the thresholds; random or
                average.
        """
        super().__init__()

        self._thresholds = {
            'random': self._random_threshold,
            'average': self._average_threshold,
        }

        assert epochs > 0, 'epochs needs to be > 0'
        assert pairs_factor > 0, 'choose a pairs factor > 0'
        assert threshold_type in self._thresholds.keys()

        self.wrapped_model = wrapped_model
        self.epochs = epochs
        self.pairs_factor = pairs_factor
        self.threshold_type = threshold_type
        self.pair_strategy = pair_strategy
        self.pair_encoding = pair_encoding
        self.threshold_sample_training = threshold_sample_training

    def fit(self, data, target, epochs=None):
        """Wraps the fit of the wrapped model.

        Args:
            data (array-like): the features.
            target (array-like): the targets.
            epochs (int, optional): required to fit the api used for
                evaluation. The value passed to the wrapped model. Default is
                None; this means the number of set epochs is used.
        """
        if epochs is None:
            epochs = self.epochs

        num_of_pairs = int(len(data) * self.pairs_factor)
        transformer = PairwiseTransformer(num_of_pairs=num_of_pairs,
                                          strategy=self.pair_strategy,
                                          pair_encoding=self.pair_encoding)
        data_trans, target_trans = transformer.fit_transform_data(data, target)

        self.wrapped_model.fit(data_trans, target_trans)

        self._fit_threshold_samples(data, target)

        if self.threshold_sample_training:
            self._train_with_threshold_samples()

    def predict(self, data):
        """Wraps the prediction and converts the task.

        Args:
            data (array-like): the features to predict labels for.
        """
        assert self._threshold_samples is not None, 'Fit the model first'

        predictions = []
        for sample in self._threshold_samples:
            references = np.tile(sample, (len(data), 1))

            if self.pair_encoding == 'concat':
                col_data = np.column_stack((references, data))
            elif self.pair_encoding == 'delta':
                col_data = references - data
            else:
                raise AssertionError(
                    f'pair_encoding {self.pair_encoding} unknown.')

            prediction = self.wrapped_model.predict(col_data)
            predictions.append(prediction >= 0)

        predictions.append(np.full_like(prediction, True))

        predictions = np.column_stack(predictions)
        predictions = np.argmax(predictions, axis=1)
        predictions = convert_array_to_class_vector(
            predictions,
            list(range(len(self._threshold_samples) + 1)),
            strategy='one_hot',
        )

        return predictions

    def _fit_threshold_samples(self, data, target):
        self._threshold_samples = []
        for c in range(target.shape[1] - 1):
            indices = np.nonzero(target[:, c] == 1)[0]
            assert len(indices) > 0, f'no threshold sample found for class {c}'

            threshold = self._thresholds[self.threshold_type](data, indices)
            self._threshold_samples.append(threshold)

    def _random_threshold(self, data, indices):
        return data[random.choice(indices)]

    def _average_threshold(self, data, indices):
        return np.average(data[indices], axis=0)

    def _train_with_threshold_samples(self):
        len_factor = 10
        t1 = []
        t2 = []
        labels = []

        t1 += self._threshold_samples[:-1]
        t2 += self._threshold_samples[1:]
        labels += ([-1] * (len(self._threshold_samples) - 1))

        t1 += self._threshold_samples
        t2 += self._threshold_samples
        labels += ([0] * len(self._threshold_samples))

        t1 += self._threshold_samples[1:]
        t2 += self._threshold_samples[:-1]
        labels = labels + ([1] * (len(self._threshold_samples) - 1))

        t1 *= len_factor
        t2 *= len_factor
        labels *= len_factor

        data = np.column_stack((t1, t2))
        self.wrapped_model.fit(data, np.array(labels))
