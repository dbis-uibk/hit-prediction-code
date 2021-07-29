"""Tests the Yang hit score function."""
import numpy as np

from hit_prediction_code.transformers.label import yang_hit_score


def test_computation():
    """Tests if the score is computed correctly."""
    pc_init = np.array(range(20))
    pc = np.expm1(pc_init)

    lc_init = np.array(range(0, 40, 2))
    lc = np.expm1(lc_init)
    score = yang_hit_score(play_count=pc, listener_count=lc)

    expected_score = pc_init * lc_init

    assert np.any(score == expected_score)
