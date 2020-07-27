"""Module to handle billboard data."""
import os.path

import pandas as pd


def read_billboard_hot_100(project_home='.'):
    """Loads all known billboard hot 100 charst.

    Args:
        project_home: path to the data folder.
    """
    bb_path = [
        project_home,
        'data',
        'billboard',
        'billboard_hot-100_1958-08-11_2019-07-06.csv',
    ]
    bb_path = os.path.join(*bb_path)

    return pd.read_csv(bb_path, header=0, index_col=0)
