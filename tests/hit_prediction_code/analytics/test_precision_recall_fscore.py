"""Test precision_recall_fscore function for multilabel confusion matrix."""
import numpy as np
import pytest
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

from hit_prediction_code.analytics import \
    confusion_matrix_to_multilabel_confusion_matrix
from hit_prediction_code.analytics import precision_recall_fscore


def test_matrix_dim_check():
    """Tests if the matrix dimension is checked."""
    with pytest.raises(AssertionError):
        precision_recall_fscore(np.zeros((5, 2, 2, 1)))


def test_matrix_shape_check():
    """Tests if it is checked that the matrix has a correct shape."""
    with pytest.raises(AssertionError):
        precision_recall_fscore(np.zeros((5, 2, 1)))


def test_average_check():
    """Tests if the average method is checked."""
    with pytest.raises(ValueError):
        precision_recall_fscore(np.zeros((5, 2, 2)), average='unknown')


def test_micro_average_score():
    """Tests if the micro average scores are correct."""
    y_true = ['cat', 'ant', 'cat', 'cat', 'ant', 'bird']
    y_pred = ['ant', 'ant', 'cat', 'cat', 'ant', 'cat']
    cm = confusion_matrix(y_true, y_pred, labels=['ant', 'bird', 'cat'])
    mcm = confusion_matrix_to_multilabel_confusion_matrix(cm)

    expected_outcome = precision_recall_fscore_support(
        y_true,
        y_pred,
        average='micro',
    )

    actual_outcome = precision_recall_fscore(mcm, average='micro')

    # precision
    assert actual_outcome[0] == expected_outcome[0]
    # recall
    assert actual_outcome[1] == expected_outcome[1]
    # f1
    assert actual_outcome[2] == expected_outcome[2]


def test_macro_average_score():
    """Tests if the macro average scores are correct."""
    y_true = ['cat', 'ant', 'cat', 'cat', 'ant', 'bird']
    y_pred = ['ant', 'ant', 'cat', 'cat', 'ant', 'cat']
    cm = confusion_matrix(y_true, y_pred, labels=['ant', 'bird', 'cat'])
    mcm = confusion_matrix_to_multilabel_confusion_matrix(cm)

    expected_outcome = precision_recall_fscore_support(
        y_true,
        y_pred,
        average='macro',
    )

    actual_outcome = precision_recall_fscore(mcm)

    # precision
    assert actual_outcome[0] == expected_outcome[0]
    # recall
    assert actual_outcome[1] == expected_outcome[1]
    # f1
    assert actual_outcome[2] == expected_outcome[2]
