"""Tests for the compute_hit_score_on_df function."""
import numpy as np
import pandas as pd

from hit_prediction_code.transformers.label import compute_hit_score_on_df


def _fake_dataframe(pc='pc', lc='lc'):
    df = pd.DataFrame()
    pc_init = np.array(range(20))
    df[pc] = np.expm1(pc_init)

    lc_init = np.array(range(0, 40, 2))
    df[lc] = np.expm1(lc_init)

    expected_score = pc_init * lc_init
    return df, expected_score


def test_compution():
    """Tests if the dataframe is correctly extended."""
    data, expected_score = _fake_dataframe()

    compute_hit_score_on_df(df=data, pc_column='pc', lc_column='lc')

    assert np.any(data['yang_hit_score'] == expected_score)


def test_compution_with_default_values():
    """Tests if the dataframe is correctly extended."""
    data, expected_score = _fake_dataframe()

    compute_hit_score_on_df(
        df=data,
        pc_column='pc',
        lc_column='lc',
        hit_score_column='yhs',
    )

    assert np.any(data['yhs'] == expected_score)
