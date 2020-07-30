"""Contains funktions to match datasets."""
import re

from logzero import logger
import pandas as pd


def match_exact_msd_bb(msd_data, bb_data):
    """Matches artist and title and only considers exact matches.

    Args:
        msd_data: dataframe containing the millionsong dataset.
        bb_data: dataframe containing the billboard hot 100.

    Returns a dataframe with the matched data.
    """
    return msd_data.merge(bb_data, on=['artist', 'title'])


def match_clean_msd_bb(msd_data, bb_data):
    """Matches artist and title after cleaning them.

    To clean artist and title the following steps are applied:
        * convert them to lower case.
        * strip whitespace at the beginning and at the end.
        * remove all char that are not alpha numerical or spaces.
        * replace multiple subsequent spaces by a single space.

    Args:
        msd_data: dataframe containing the millionsong dataset.
        bb_data: dataframe containing the billboard hot 100.

    Returns a dataframe with the matched data.
    """

    def clean_column(value):
        value = str(value)

        value = value.strip()
        value = value.lower()
        value = re.sub(r'[^a-zA-Z0-9\s]', '', value)
        value = re.sub(r'\s+', ' ', value)

        if value == '':
            value = None

        return value

    bb_data['artist_clean'] = bb_data['artist'].apply(clean_column)
    msd_data['artist_clean'] = msd_data['artist'].apply(clean_column)

    bb_data['title_clean'] = bb_data['title'].apply(clean_column)
    msd_data['title_clean'] = msd_data['title'].apply(clean_column)

    bb_len = len(bb_data)
    bb_data = bb_data.drop_duplicates(
        subset=['artist_clean', 'title_clean'],
        keep=False,
    )
    if len(bb_data) < bb_len:
        logger.warning('Removed duplicates from billboard data')

    msd_bb = msd_data.merge(
        bb_data,
        on=['artist_clean', 'title_clean'],
        suffixes=('_msd', '_bb'),
    )
    msd_bb.dropna(subset=['artist_clean', 'title_clean'], inplace=True)

    return msd_bb


def filter_duplicates(data, id_cols, target_col, keep_lowest):
    """Filters duplicates based on the target value.

    Args:
        data: dataframe containing all samples.
        id_cols: list of columns identifying a song.
        target: target to consider.
        keep_lowest: True, if all entries containing a minimum value are keep.
            Otherwise, the max value is kept.

    Returns the filtered dataset.
    """
    keep = []
    for _, group in data.groupby(id_cols):
        if keep_lowest is True:
            keep_group = group[group[target_col] <= group[target_col].min()]
        else:
            keep_group = group[group[target_col] >= group[target_col].max()]

        keep.append(keep_group)

    return pd.concat(keep)
