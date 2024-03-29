"""Wrapper implementation for SVM."""
import numpy as np
from sklearn import svm

from ..transformers.label import convert_array_to_class_vector


class SVR(svm.SVR):
    """Wrapper class for the sklearn SVR."""

    def __init__(self):
        """Creates the wrapped SVR."""
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


class SVC(svm.SVC):
    """Wrapper class for the sklearn SVC."""

    def __init__(self):
        """Creates the wrapped SVC."""
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
