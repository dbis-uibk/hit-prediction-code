"""Tests the scores_from_multilabel_confusion_matrix function."""
import numpy as np

from hit_prediction_code.analytics import \
    confusion_matrix_to_multilabel_confusion_matrix
from hit_prediction_code.analytics import \
    scores_from_multilabel_confusion_matrix


def test_score_computation():
    """Tests if the scores are computed as expected."""
    mcm = confusion_matrix_to_multilabel_confusion_matrix(
        np.arange(16).reshape(4, 4))

    macro_p, micro_p = (.22693452380952384, .25)
    macro_r, micro_r = (.19205209994683678, .25)
    macro_f1, micro_f1 = (.20476190476190476, .25)
    expected_outcome = {
        'macro_precision': macro_p,
        'micro_precision': micro_p,
        'macro_recall': macro_r,
        'micro_recall': micro_r,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
    }
    actual_outcome = scores_from_multilabel_confusion_matrix(mcm)

    assert actual_outcome == expected_outcome
