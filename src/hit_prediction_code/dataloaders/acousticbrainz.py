"""Module to handle acousticbrainz data."""
import glob
import json
import os

from logzero import logger
import pandas as pd


def load_ab_features_as_df(mbids, feature_type, project_home='.'):
    """Loads highlevel or lowlevel features from acousticbrainz.

    Args:
        mbids: a list of musicbrainz ids.
        feature_type: use highlevel (hl) or lowlevel (ll).
        project_home: path to the data folder

    Returns a dataframe.
    """
    features = []
    for item in load_ab_features(mbids=mbids,
                                 feature_type=feature_type,
                                 project_home=project_home):
        features += list(item)
    return pd.json_normalize(features)


def load_ab_features(mbids, feature_type, project_home='.'):
    """Loads highlevel or lowlevel features from acousticbrainz.

    Args:
        mbids: a list of musicbrainz ids.
        feature_type: use highlevel (hl) or lowlevel (ll).
        project_home: path to the data folder

    Returns a generator of generators generating dicts.
    """
    if feature_type in ['hl', 'highlevel']:
        for mbid in mbids:
            yield _get_ab_features(
                mbid=mbid,
                project_home=project_home,
                path_type='highlevel',
            )
    elif feature_type in ['ll', 'lowlevel']:
        for mbid in mbids:
            yield _get_ab_features(
                mbid=mbid,
                project_home=project_home,
                path_type='lowlevel',
            )
    else:
        raise ValueError('feature_type unknown.')


def _get_ab_features(mbid, project_home, path_type):
    data_path = [project_home, 'data', 'acousticbrainz', 'processed']
    path = os.path.join(*data_path, path_type, mbid[0], mbid[0:2], mbid)

    search_path = path + '-*.json'
    logger.debug('Search for: \'%s\'' % search_path)
    for filename in sorted(glob.glob(search_path)):
        logger.debug('Load: %s' % filename)
        with open(filename, 'r') as stream:
            feature_data = json.load(stream)
            feature_data['mbid'] = mbid
            feature_data['file'] = filename[len(project_home) + 1:]

            yield feature_data
