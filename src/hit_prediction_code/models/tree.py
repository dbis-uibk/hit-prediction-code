"""Wrapper implementations of tree based models."""
import numpy as np
from sklearn import ensemble

from ..transformers.label import convert_array_to_class_vector


class RandomForestRegressor(ensemble.RandomForestRegressor):
    """Wrapper class for the sklearn random forest regressor."""

    def __init__(self):
        """Creates the wrapped random forest regressor."""
        super().__init__(verbose=True)
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


class RandomForestClassifier(ensemble.RandomForestClassifier):
    """Wrapper class for the sklearn random forest classifier."""

    def __init__(self):
        """Creates the wrapped random forest classifier."""
        super().__init__(verbose=True)
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
