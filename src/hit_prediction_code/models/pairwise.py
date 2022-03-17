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

    def __init__(self, wrapped_model: BaseEstimator, epochs: int = 1) -> None:
        """Creates the wrapper.

        Args:
            wrapped_model (BaseEstimator): the model used for the actual
                prediction.
        """
        super().__init__()

        assert epochs > 0, 'epochs needs to be > 0'

        self.wrapped_model = wrapped_model
        self.epochs = epochs

    def fit(self, data, target, epochs=None):
        """Wraps the fit of the wrapped model.

        Args:
            data (array-like): the features.
            target (array-like): the targets.
            epochs (int, optional): required to fit the api used for
                evaluation. The value passed to the wrapped model. Default is
                None; this means the number of set epochs is used.
        """
        assert target.shape[1] == 3, 'Only three classes are supported.'

        if epochs is None:
            epochs = self.epochs

        self._fit_threshold_sample(data, target)

        transformer = PairwiseTransformer(num_of_pairs=len(data))
        data, target = transformer.fit_transform_data(data, target)

        self.wrapped_model.fit(data, target, epochs=epochs)

    def predict(self, data):
        """Wraps the prediction and converts the task.

        Args:
            data (array-like): the features to predict labels for.
        """
        assert self._threshold_sample is None, 'Fit the model first'

        references = np.tile(self._threshold_sample, (len(data), 1))
        data = np.column_stack((references, data))

        prediction = self.wrapped_model.predict(data)
        prediction = convert_array_to_class_vector(
            prediction,
            [-1, 0, 1],
            strategy='one_hot',
        )

        return prediction

    def _fit_threshold_sample(self, data, target):
        indices = np.nonzero(target[:, 1] == 1)[0]

        assert len(indices) > 0, 'no threshold sample found'

        self._threshold_sample = data[random.choice(indices)]
