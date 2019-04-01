"""Dataloaders for the hit song prediction."""
import re
import json

from dbispipeline.base import Loader

import numpy as np

import pandas as pd


class MsdBbLoader(Loader):
    """Million song dataset / billboard charts loaer."""

    def __init__(self,
                 hits_file_path,
                 non_hits_file_path,
                 features_path,
                 non_hits_per_hit=None,
                 features=None,
                 label=None):
        self._config = {
            'hits_file_path': hits_file_path,
            'non_hits_file_path': non_hits_file_path,
            'features_path': features_path,
            'non_hits_per_hit': non_hits_per_hit,
            'features': features,
            'label': label,
        }

        hits = pd.read_csv(hits_file_path)
        non_hits = pd.read_csv(non_hits_file_path)

        if non_hits_per_hit:
            non_hits = non_hits.sample(n=len(hits) * non_hits_per_hit)

        data = hits.append(non_hits, sort=False, ignore_index=True)
        ll_features = pd.read_hdf(features_path + '/msd_bb_ll_features.h5')
        data = data.merge(ll_features, on='msd_id')
        hl_features = pd.read_hdf(features_path + '/msd_bb_hl_features.h5')
        data = data.merge(hl_features, on='msd_id')

        self.labels = data[[label]]

        non_label_columns = list(data.columns)
        non_label_columns.remove(label)
        data = data[non_label_columns]

        feature_data = []
        for feature in features:
            if feature == 'hl':
                regex_filter = r'highlevel\.\w+\.all\.\w+'
            else:
                regex_filter = feature

            feature_data.append(data[_filter_features(data.columns,
                                                      regex_filter)])

        self.data = np.array(feature_data)

    def load(self):
        return self.data, self.labels

    def configuration(self):
        return self._config


def _get_highlevel_feature(features_path, msd_id):
    file_suffix = '.mp3.highlevel.json'
    return _load_feature(features_path, msd_id, file_suffix)


def _get_lowlevel_feature(features_path, msd_id):
    file_suffix = '.mp3'
    return _load_feature(features_path, msd_id, file_suffix)


def _load_feature(features_path, msd_id, file_suffix):
    file_prefix = '/features_tracks_' + msd_id[2].lower() + '/'
    file_name = features_path + file_prefix + msd_id + file_suffix

    with open(file_name) as features:
        return json.load(features)


def _filter_features(columns, regex_filter):
    regex = re.compile(regex_filter)
    return filter(regex.search, columns)
