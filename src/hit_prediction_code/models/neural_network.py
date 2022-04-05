"""Wrapper implementation for MLP."""
import numpy as np
from sklearn import neural_network

from ..transformers.label import convert_array_to_class_vector


class MLPRegressor(neural_network.MLPRegressor):
    """Wrapper class for the sklearn MLPRegressor."""

    def __init__(self, hidden_layer_sizes=(100,), max_iter=200, verbose=False):
        """Creates the wrapped MLPRegressor."""
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            verbose=verbose,
        )
        self.epochs = 1

    def fit(self, data, target, epochs=1):
        """Wraps the fit of the super class.

        This allows to use this class in an epoch evaluator. Keep in mind that
        the number of epochs is ignored. Hence, it should only be used with 1
        epoch.

        Args:
            data (array-like): the features.
            target (array-like): the targets.
            epochs (int, optional): required to fit the api used for
                evaluation. The value is ignored and it is always trained for 1
                epoch.
        """
        super().fit(data, target)


class MLPClassifier(neural_network.MLPClassifier):
    """Wrapper class for the sklearn MLPClassifier."""

    def __init__(self, hidden_layer_sizes=(100,), max_iter=200, verbose=False):
        """Creates the wrapped MLPClassifier."""
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            verbose=verbose,
        )
        self._num_classes = 0
        self.epochs = 1

    def fit(self, data, target, epochs=1):
        """Wraps the fit of the super class.

        This allows to use this class in an epoch evaluator. Keep in mind that
        the number of epochs is ignored. Hence, it should only be used with 1
        epoch.

        Args:
            data (array-like): the features.
            target (array-like): the targets.
            epochs (int, optional): required to fit the api used for
                evaluation. The value is ignored and it is always trained for 1
                epoch.
        """
        self._num_classes = target.shape[1]
        target = np.argmax(target, axis=1)
        super().fit(data, target)

    def predict(self, x):
        """Wraps the predict and converts integers to classes."""
        prediction = super().predict(x)

        return convert_array_to_class_vector(
            prediction,
            labels=list(range(self._num_classes)),
            strategy='one_hot',
        )
