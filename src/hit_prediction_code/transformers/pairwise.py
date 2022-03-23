"""Module implementing a pairwise transformer."""
import random
from typing import Any, List, Tuple

import numpy as np
from numpy.typing import ArrayLike


class PairwiseTransformer(object):
    """Custom transformer able to transform data and labels."""

    def __init__(self,
                 num_of_pairs: int,
                 strategy: str = 'random',
                 pair_encoding: str = 'concat') -> None:
        """Creates the transformer.

        Args:
            num_of_pairs (int): the number of created pairs
            strategy (str, optional): Determines how pairs are sampled.
                Defaults to 'random'.
        """
        self._num_of_pairs = num_of_pairs

        self._strategies = {
            'random': self._create_random_pairs,
            'balanced': self._create_balanced_pairs,
        }

        strategies = self._strategies.keys()
        error_msg = f'Strategy {strategy} not supported. Use: {strategies}'
        assert strategy in strategies, error_msg
        self._strategy = strategy

        self._pair_encodings = {
            'concat': create_concat_pairs,
            'delta': create_delta_pairs,
        }

        encodings = self._pair_encodings.keys()
        error_msg = f'Encoding {pair_encoding} not supported. Use: {encodings}'
        assert pair_encoding in encodings, error_msg
        self._pair_encoding = pair_encoding

        self._fitted_data_shape = None
        self._fitted_labels_shape = None

    def fit(self, data: ArrayLike, labels: List[int], **fit_params) -> Any:
        """Fits the transformer.

        Args:
            data (ArrayLike): the data to fit.
            labels (List[int]): the labels to fit. Needs to be 2D where rows
                contain samples and columns represent classes.

        Returns:
            PairwiseTransformer: the fitted transformer.
        """
        num_samples = len(data)
        num_labels = len(labels)
        assert num_samples == num_labels, 'Data and labels need same length.'

        assert len(labels.shape) == 2, 'Labels need to be a 2D array.'

        num_pairs = num_samples * num_samples

        assert num_pairs >= self._num_of_pairs, 'Not enough possible pairs.'

        self._data_shape = data.shape
        self._labels_shape = labels.shape

        return self

    def transform_data(self, data: ArrayLike,
                       labels: List[int]) -> Tuple[ArrayLike, ArrayLike]:
        """Performs the transformation.

        Args:
            data (ArrayLike): the data to transform.
            labels (List[int]): the labels to transform.

        Returns:
            Tuple[ArrayLike, ArrayLike]: the first element contains the
                transformed data and the second element contains the
                transformed labels.
        """
        assert self._data_shape is not None, 'Transformer not fitted.'

        data_shape_fits = (len(self._data_shape) == len(data.shape) and
                           self._data_shape[1] == data.shape[1])
        assert data_shape_fits, 'Data shape does not fit.'

        labels_shape_fits = (len(self._labels_shape) == len(labels.shape) and
                             self._labels_shape[1] == labels.shape[1])
        assert labels_shape_fits, 'Labels shape does not fit.'

        num_samples = len(data)
        num_labels = len(labels)
        assert num_samples == num_labels, 'Data and labels need same length.'

        labels = np.argmax(labels, axis=1)

        random1, random2, pair_labels = self._strategies[self._strategy](
            data,
            labels,
        )

        pair_labels = list(_compute_labels(labels, random1, random2))
        pair_labels = np.array(pair_labels)

        pair_data = self._pair_encodings[self._pair_encoding](data[random1],
                                                              data[random2])
        return pair_data, pair_labels

    def fit_transform_data(self, data: ArrayLike,
                           labels: List[int]) -> Tuple[ArrayLike, ArrayLike]:
        """Fits and transforms in one operation.

        Args:
            data (ArrayLike): the data to fit and transform.
            labels (List[int]): the labels to fit an transform.

        Returns:
            Tuple[ArrayLike, ArrayLike]: the first element contains the[1:]
                transformed data and the second element contains the
                transformed labels.
        """
        return self.fit(data, labels).transform_data(data, labels)

    def _create_random_pairs(self, data, labels):
        random1 = random.choices(range(len(data)), k=self._num_of_pairs)
        random2 = random.choices(range(len(data)), k=self._num_of_pairs)
        pair_labels = list(_compute_labels(labels, random1, random2))

        return random1, random2, pair_labels

    def _create_balanced_pairs(self, data, labels):
        random1 = []
        random2 = []
        pair_labels = []

        label = 1
        while len(random1) < self._num_of_pairs:
            for expected_label in [-1, 0, 1]:
                while label != expected_label:
                    item1 = random.randrange(len(data))
                    item2 = random.randrange(len(data))
                    label = _compare_labels_by_index(labels, item1, item2)

                random1.append(item1)
                random2.append(item2)
                pair_labels.append(label)

                if len(random1) >= self._num_of_pairs:
                    break

        return random1, random2, pair_labels


def _compute_labels(labels, choice1, choice2):
    assert len(choice1) == len(choice2)

    for index1, index2 in zip(choice1, choice2):
        yield _compare_labels_by_index(labels, index1, index2)


def _compare_labels_by_index(labels, index1, index2):
    lx = labels[index1]
    ly = labels[index2]

    if lx < ly:
        return -1
    elif lx == ly:
        return 0
    else:  # lx > ly
        return 1


def create_concat_pairs(choice1, choice2):
    """Creates concatenated features from pairs."""
    return np.column_stack((choice1, choice2))


def create_delta_pairs(choice1, choice2):
    """Creates delta encoding from pairs."""
    return choice1 - choice2
