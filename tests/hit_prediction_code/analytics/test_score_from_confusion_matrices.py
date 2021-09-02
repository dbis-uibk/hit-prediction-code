"""Tests the scores_from_confusion_matrices function."""
import numpy as np
import pandas as pd
import pytest

from hit_prediction_code.analytics import scores_from_confusion_matrices


def test_array_length_check():
    """Tests if it is ensured that both arrays have the same size."""
    with pytest.raises(AssertionError):
        scores_from_confusion_matrices([None, None], [None, None, None])


def test_score_computation():
    """Tests if the scores are computed as expected."""
    epochs = [1, 2, 3]
    cms = [
        np.arange(16).reshape(4, 4),
        np.arange(16).reshape(4, 4),
        np.arange(16).reshape(4, 4),
    ]

    macro_p, micro_p = (.22693452380952384, .25)
    macro_r, micro_r = (.19205209994683678, .25)
    macro_f1, micro_f1 = (.20476190476190476, .25)
    expected_outcome = pd.DataFrame([
        {
            'epochs': 1,
            'macro_precision': macro_p,
            'micro_precision': micro_p,
            'macro_recall': macro_r,
            'micro_recall': micro_r,
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
        },
        {
            'epochs': 2,
            'macro_precision': macro_p,
            'micro_precision': micro_p,
            'macro_recall': macro_r,
            'micro_recall': micro_r,
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
        },
        {
            'epochs': 3,
            'macro_precision': macro_p,
            'micro_precision': micro_p,
            'macro_recall': macro_r,
            'micro_recall': micro_r,
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
        },
    ])

    actual_outcome = scores_from_confusion_matrices(cms, epochs)

    pd.testing.assert_frame_equal(actual_outcome, expected_outcome)
