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

    data = np.arange(10)

    transformer = PairwiseTransformer(number_of_pairs, strategy='random')

    # labels and data need to be the same to fullfil the assumption of the
    # _assert_label_computation()
    transformed_data, transformed_labels = transformer.fit_transform_data(
        np.column_stack((data, data)),
        data,
    )

    assert transformed_data.shape == (number_of_pairs, 4)
    assert transformed_labels.shape == (number_of_pairs, 1)

    _assert_label_computation(transformed_data, transformed_labels)


def test_strategy_balanced():
    """Tests balanced combination of pairs."""
    number_of_pairs = 12

    data = np.arange(10)

    transformer = PairwiseTransformer(number_of_pairs, strategy='balanced')

    # labels and data need to be the same to fullfil the assumption of the
    # _assert_label_computation()
    transformed_data, transformed_labels = transformer.fit_transform_data(
        np.column_stack((data, data)),
        data,
    )

    assert transformed_data.shape == (number_of_pairs, 4)
    assert transformed_labels.shape == (number_of_pairs, 1)

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
