"""Module implementing a pairwise transformer."""
import random
from typing import Any, List, Tuple

import numpy as np
from numpy.typing import ArrayLike


class PairwiseTransformer(object):
    """Custom transformer able to transform data and labels."""

    def __init__(self, num_of_pairs: int, strategy: str = 'random') -> None:
        """Creates the transformer.

        Args:
            num_of_pairs (int): the number of created pairs
            strategy (str, optional): Determines how pairs are sampled.
                Defaults to 'random'.
        """
        self._strategies = {
            'random': self._create_random_pairs,
            'balanced': self._create_balanced_pairs,
        }

        strategies = self._strategies.keys()

        error_msg = f'Strategy {strategy} not supported. Use: {strategies}'
        assert strategy in strategies, error_msg

        self._num_of_pairs = num_of_pairs
        self._strategy = strategy

    def fit(self, data: ArrayLike, labels: List[int], **fit_params) -> Any:
        """Fits the transformer.

        Args:
            data (ArrayLike): the data to fit.
            labels (List[int]): the labels to fit.

        Returns:
            PairwiseTransformer: the fitted transformer.
        """
        assert len(data) * len(
            data) >= self._num_of_pairs, 'Not enough possible combinations.'

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
        random1, random2, pair_labels = self._strategies[self._strategy](
            data,
            labels,
        )

        pair_labels = list(_compute_labels(labels, random1, random2))
        pair_labels = np.array(pair_labels).reshape(self._num_of_pairs, 1)

        return np.column_stack((data[random1], data[random2])), pair_labels

    def fit_transform_data(self, data: ArrayLike,
                           labels: List[int]) -> Tuple[ArrayLike, ArrayLike]:
        """Fits and transforms in one operation.

        Args:
            data (ArrayLike): the data to fit and transform.
            labels (List[int]): the labels to fit an transform.

        Returns:
            Tuple[ArrayLike, ArrayLike]: the first element contains the
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
