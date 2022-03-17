"""Tests if the pairwise transformer correctly transforms a dataframe."""
import numpy as np
import pytest

from hit_prediction_code.transformers.pairwise import PairwiseTransformer


def _assert_label_computation(data, labels):
    """Assumes that data and labels have the same order before transformation.

    Further, we assume, that if there is more than one feature column, that we
    all features are the same.
    """
    assert len(data) == len(labels)

    for feature, label in zip(data, labels):
        if feature[0] < feature[-1]:
            assert label == -1
        elif feature[0] == feature[-1]:
            assert label == 0
        else:  # feature[0] > feature[-1]
            assert label == 1


def test_strategy_assert():
    """Tests if the strategies are checked."""
    with pytest.raises(AssertionError):
        PairwiseTransformer(10, strategy='unknown_strategy')


def test_strategy_random():
    """Tests random combination of pairs."""
    number_of_pairs = 12
    number_of_samples = 10

    data = np.arange(number_of_samples)
    data = np.column_stack((data, data))
    data = np.concatenate((data, data), axis=0)

    labels = np.identity(number_of_samples)
    labels = np.concatenate((labels, labels), axis=0)

    transformer = PairwiseTransformer(number_of_pairs, strategy='random')

    # labels and data need to be the same to fullfil the assumption of the
    # _assert_label_computation()
    transformed_data, transformed_labels = transformer.fit_transform_data(
        data,
        labels,
    )

    assert transformed_data.shape == (number_of_pairs, data.shape[1] * 2)
    assert transformed_labels.shape == (number_of_pairs,)

    _assert_label_computation(transformed_data, transformed_labels)


def test_strategy_balanced():
    """Tests balanced combination of pairs."""
    number_of_pairs = 12
    number_of_samples = 10

    data = np.arange(number_of_samples)
    data = np.column_stack((data, data))
    data = np.concatenate((data, data), axis=0)

    labels = np.identity(number_of_samples)
    labels = np.concatenate((labels, labels), axis=0)

    transformer = PairwiseTransformer(number_of_pairs, strategy='balanced')

    # labels and data need to be the same to fullfil the assumption of the
    # _assert_label_computation()
    transformed_data, transformed_labels = transformer.fit_transform_data(
        data,
        labels,
    )

    assert transformed_data.shape == (number_of_pairs, data.shape[1] * 2)
    assert transformed_labels.shape == (number_of_pairs,)

    _assert_label_computation(transformed_data, transformed_labels)

    smaller = 0
    equal = 0
    bigger = 0
    for label in transformed_labels:
        if label == -1:
            smaller += 1
        elif label == 0:
            equal += 1
        else:  # label == 1
            bigger += 1

    assert smaller == equal
    assert equal == bigger
