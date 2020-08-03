"""Contains funktions to match datasets."""
import re
import uuid

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


def clean_value(value):
    """Cleans the value content.

    To clean a value the following steps are applied:
        * convert them to lower case.
        * strip whitespace at the beginning and at the end.
        * remove all char that are not alpha numerical or spaces.
        * replace multiple subsequent spaces by a single space.

    Args:
        value: the value to clean.
    """
    value = str(value)

    value = value.strip()
    value = value.lower()
    value = re.sub(r'[^a-zA-Z0-9\s]', '', value)
    value = re.sub(r'\s+', ' ', value)

    if value == '':
        value = None

    return value


def clean_artist_title(data, artist_col='artist', title_col='title'):
    """Cleans artist and title column.

    Args:
        data: dataframe containing all songs.
        artist_col: column name containing the artist.
        title_col: column name containing the title.

    Returns the datarame contained cleaned artist and title.
    """
    data[artist_col + '_clean'] = data[artist_col].apply(clean_value)
    data[title_col + '_clean'] = data[title_col].apply(clean_value)

    data_len = len(data)
    data.dropna(subset=['artist_clean', 'title_clean'], inplace=True)
    if len(data) < data_len:
        logger.warning(
            'Removed rows where artist_clean or title_clean is empty.')

    return data


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
    bb_data = clean_artist_title(bb_data)
    msd_data = clean_artist_title(msd_data)

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


def add_uuid_column(data,
                    artist_col='artist_clean',
                    title_col='title_clean',
                    furhter_id_cols='default'):
    """Assigns a UUID for songs.

    First, a unique uuid is assigned to song identified by artist_col and
    title_col. After that each key in the list furhter_id_cols is used to merge
    duplicates and assign a single id to them.

    Args:
        data: dataframe containing all songs.
        artist_col: column name containing the artist.
        title_col: column name containing the title.
        furhter_id_cols: list of columns containing further ids where the uuid
            has to be unique. 'default' is ['mbid', 'msd_id', 'echo_nest_id']

    Retruns a dataframe extended by a UUID.
    """
    if furhter_id_cols == 'default':
        furhter_id_cols = ['mbid', 'msd_id', 'echo_nest_id']

    join_cols = [artist_col, title_col]
    songs = data[join_cols].drop_duplicates()

    # add random uuid based on join cols
    for idx, _ in songs.iterrows():
        songs.loc[idx, 'uuid'] = uuid.uuid4()

    data = data.merge(songs, on=join_cols)

    def unite_uuid(id_cols):
        for _, group in data.groupby(id_cols):
            if group['uuid'].nunique() > 1:
                selected_uuid = group.iloc[0]['uuid']
                for row_uuid in group['uuid'].unique():
                    if row_uuid != selected_uuid:
                        data.loc[data.uuid == row_uuid, 'uuid'] = selected_uuid

    for col in furhter_id_cols:
        unite_uuid(col)

    return data
