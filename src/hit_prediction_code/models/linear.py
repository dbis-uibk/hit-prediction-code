"""Wrapper implementations of linear models."""
import numpy as np
from sklearn import linear_model

from ..transformers.label import convert_array_to_class_vector


class LinearRegression(linear_model.LinearRegression):
    """Wrapper class for the sklearn linear regression regressor."""

    def __init__(self):
        """Creates the wrapped regressor."""
        super().__init__()
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


class LogisticRegression(linear_model.LogisticRegression):
    """Wrapper class for the sklearn logistic regression regressor."""

    def __init__(self):
        """Creates the wrapped logistic regression regressor."""
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


class LogisticRegressionClassifier(linear_model.LogisticRegression):
    """Wrapper class for the sklearn logistic regression classifier."""

    def __init__(self):
        """Creates the wrapped logistic regression classifier."""
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
