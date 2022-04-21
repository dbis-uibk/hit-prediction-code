"""Module containing implementations for ordinal classification."""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import clone

from ..transformers.label import convert_array_to_class_vector


class OrdinalClassifier():
    """Ordinal classifier implementation.

    Taken from: https://gist.github.com/M46F/c574f688715d5f7e4b65bce4b3ec5fdc
    """

    def __init__(self, clf, label_mapper):
        """Initializes the classifier.

        Args:
            clf: base classifier.
            label_mapper: function used to map labels.
        """
        self.clf = clf
        self.clfs = {}
        self.label_mapper = label_mapper
        self.epochs = 1

    def fit(self, data, labels, epochs=None):
        """Fits the classifier."""
        labels = self.label_mapper(labels)
        self.unique_class = np.sort(np.unique(labels))
        if self.unique_class.shape[0] > 2:
            for i in range(self.unique_class.shape[0] - 1):
                # for each k - 1 ordinal value we fit a binary classifier
                binary_y = (labels > self.unique_class[i]).astype(np.uint8)
                clf = clone(self.clf)
                clf.fit(data, binary_y)
                self.clfs[i] = clf

    def predict_proba(self, data):
        """Predicts label probabilities."""
        clfs_predict = {k: self.clfs[k].predict_proba(data) for k in self.clfs}
        predicted = []
        for i, y in enumerate(self.unique_class):
            if i == 0:
                # V1 = 1 - Pr(y > V1)
                predicted.append(1 - clfs_predict[y][:, 1])
            elif y in clfs_predict:
                # Vi = Pr(y > Vi-1) - Pr(y > Vi)
                predicted.append(clfs_predict[y - 1][:, 1] -
                                 clfs_predict[y][:, 1])
            else:
                # Vk = Pr(y > Vk-1)
                predicted.append(clfs_predict[y - 1][:, 1])
        return np.vstack(predicted).T

    def predict(self, data):
        """Predicts the target label."""
        return np.argmax(self.predict_proba(data), axis=1)


class RegressorOrdinalModel(ClassifierMixin, BaseEstimator):
    """This class wraps a regression model to predict ordinal classes."""

    def __init__(self, wrapped_model: BaseEstimator, epochs: int = 1) -> None:
        """Creates the wrapper.

        Args:
            wrapped_model (BaseEstimator): the model used for the actual
                prediction.
            epochs (int): number of epochs to train.
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
        if epochs is None:
            epochs = self.epochs

        self._num_classes = target.shape[1]
        target_trans = np.argmax(target, axis=1)

        self.wrapped_model.fit(data, target_trans)

    def predict(self, data):
        """Wraps the prediction and converts the task.

        Args:
            data (array-like): the features to predict labels for.
        """
        predictions = self.wrapped_model.predict(data)

        predictions = convert_array_to_class_vector(
            predictions,
            list(range(self._num_classes)),
            strategy='one_hot',
        )

        return predictions
