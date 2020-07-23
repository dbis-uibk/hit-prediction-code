"""Module to handle millionsongdataset data."""
import json
import os

from logzero import logger
import pandas as pd


def load_msd_features_as_df(msd_ids, feature_type, project_home='.'):
    """Loads highlevel or lowlevel features from millionsongdataset.

    Args:
        msd_ids: a list of millionsongdataset ids.
        feature_type: use highlevel (hl) or lowlevel (ll).
        project_home: path to the data folder

    Returns a dataframe.
    """
    features = []
    for item in load_msd_features(msd_ids=msd_ids,
                                  feature_type=feature_type,
                                  project_home=project_home):
        features += list(item)
    return pd.json_normalize(features)


def load_msd_features(msd_ids, feature_type, project_home='.'):
    """Loads highlevel or lowlevel features from millionsongdataset.

    Args:
        mbids: a list of millionsongdataset ids.
        feature_type: use highlevel (hl) or lowlevel (ll).
        project_home: path to the data folder

    Returns a generator of generators generating dicts.
    """
    if feature_type in ['hl', 'highlevel']:
        for msd_id in msd_ids:
            yield _get_msd_features(
                msd_id=msd_id,
                project_home=project_home,
                path_type='mp3.highlevel',
            )
    elif feature_type in ['ll', 'lowlevel']:
        for msd_id in msd_ids:
            yield _get_msd_features(
                msd_id=msd_id,
                project_home=project_home,
                path_type='lowlevel',
            )
    else:
        raise ValueError('feature_type unknown.')


def _get_msd_features(msd_id, project_home, path_type):
    data_path = [
        project_home,
        'data',
        'millionsongdataset',
        'msd_audio_features',
    ]
    sub_folder = 'features_tracks_' + msd_id[2].lower()
    path = os.path.join(*data_path, sub_folder, msd_id)

    filename = path + '.' + path_type + '.json'
    logger.debug('Search for: \'%s\'' % filename)
    if os.path.isfile(filename):
        logger.debug('Load: %s' % filename)
        with open(filename, 'r') as stream:
            feature_data = json.load(stream)
            feature_data['msd_id'] = msd_id
            feature_data['file'] = filename[len(project_home) + 1:]

            yield feature_data
