"""Module containing implementations for ordinal classification."""
import numpy as np
from sklearn.base import clone


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
