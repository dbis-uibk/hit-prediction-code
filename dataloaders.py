"""Dataloaders for the hit song prediction."""
import re
import json

from dbispipeline.base import Loader

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

        self.data = hits.append(non_hits, sort=False, ignore_index=True)
        ll_features = pd.read_hdf(features_path + '/msd_bb_ll_features.h5')
        self.data = self.data.merge(ll_features, on='msd_id')
        hl_features = pd.read_hdf(features_path + '/msd_bb_hl_features.h5')
        self.data = self.data.merge(hl_features, on='msd_id')

        self.labels =  self.data[[label]]
        self.data = self.data[_filter_hl_features(self.data.columns)]

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


def _filter_hl_features(columns, label):
    regex = re.compile('highlevel\.\w+\.all\.\w+')

    non_label_columns = list(columns)
    non_label_columns.remove(label)

    return filter(regex.search, non_label_columns)
