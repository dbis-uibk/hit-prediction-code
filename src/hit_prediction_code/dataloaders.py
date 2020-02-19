"""Dataloaders for the hit song prediction."""
import logging
import json

from dbispipeline.base import Loader
import numpy as np
import pandas as pd

from .common import feature_columns

LOGGER = logging.getLogger(__name__)


class MsdBbLoader(Loader):
    """Million song dataset / billboard charts loader."""

    def __init__(self,
                 hits_file_path,
                 non_hits_file_path,
                 features_path,
                 non_hits_per_hit=None,
                 features=None,
                 label=None,
                 nan_value=0,
                 random_state=None):
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
            num_of_samples = len(hits) * non_hits_per_hit
            non_hits = non_hits.sample(n=num_of_samples,
                                       random_state=random_state)

        data = hits.append(non_hits, sort=False, ignore_index=True)
        ll_features = pd.read_hdf(features_path + '/msd_bb_ll_features.h5')
        data = data.merge(ll_features, on='msd_id')
        hl_features = pd.read_hdf(features_path + '/msd_bb_hl_features.h5')
        data = data.merge(hl_features, on='msd_id')
        data = key_mapping(data)

        self.labels = np.ravel(data[[label]])
        nan_values = pd.isnull(self.labels)
        self.labels[nan_values] = nan_value

        non_label_columns = list(data.columns)
        non_label_columns.remove(label)
        data = data[non_label_columns]

        feature_cols = []
        self._features_list = []
        for feature in features:
            cols, part = feature_columns(data.columns, feature)
            feature_cols += cols
            self._features_list.append((cols, part))

        self.data = data[feature_cols]
        self._features_index_list = []
        for cols, part in self._features_list:
            index = [self.data.columns.get_loc(c) for c in cols]
            self._features_index_list.append((index, part))

        self._config['features_list'] = self._features_list
        LOGGER.info(self._features_list)

    def load(self):
        return self.data.values, self.labels

    @property
    def feature_indices(self):
        return self._features_index_list

    @property
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


def key_mapping(df):
    # https://stackoverflow.com/questions/37292872/how-can-i-one-hot-encode-in-python
    string_columns = [
        'tonal.chords_key', 'tonal.chords_scale', 'tonal.key_scale',
        'tonal.key_key', 'artist'
    ]
    for c in string_columns:
        if c in list(df):
            if c == 'artist':
                values = df[c]
                counts = pd.value_counts(values)
                mask = values.isin(counts[counts > 1].index)
                values[~mask] = 'one-hit-wonder'
                dummies = pd.get_dummies(values, prefix=c, drop_first=False)
            else:
                dummies = pd.get_dummies(df[c], prefix=c, drop_first=False)

            df = pd.concat([df.drop(c, axis=1), dummies], axis=1)
    return df


class MelSpectLoader(Loader):
    """Loads dataset with hits and non-hits contaning melspectrogramms."""

    def __init__(self, dataset_path, features=None, label=None, nan_value=0):
        self._config = {
            'dataset_path': dataset_path,
            'features': features,
            'label': label,
        }

        data = pd.read_pickle(dataset_path)

        self.labels = np.ravel(data[[label]])
        nan_values = pd.isnull(self.labels)
        self.labels[nan_values] = nan_value

        # ensure that the array is at least 2d
        if len(self.labels.shape) == 1:
            self.labels = self.labels.reshape((*self.labels.shape, 1))

        non_label_columns = list(data.columns)
        non_label_columns.remove(label)
        data = data[non_label_columns]

        # TODO: do this in a generic fashion and rename the dataloader.
        self.data = data[features].values
        self.data = np.stack(self.data, axis=0)

    def load(self):
        """Returns the data loaded by the dataloader."""
        return self.data, self.labels

    @property
    def configuration(self):
        """Returns the configuration in json serializable format."""
        return self._config


class EssentiaLoader(Loader):
    """Essentia feature loader."""

    def __init__(self, dataset_path, features=None, label=None, nan_value=0):
        self._config = {
            'dataset_path': dataset_path,
            'features': features,
            'label': label,
            'nan_value': nan_value,
        }

        data = pd.read_pickle(dataset_path)
        data = key_mapping(data)

        self.labels = np.ravel(data[[label]])
        nan_values = pd.isnull(self.labels)
        self.labels[nan_values] = nan_value

        non_label_columns = list(data.columns)
        non_label_columns.remove(label)
        data = data[non_label_columns]

        feature_cols = []
        self._features_list = []
        for feature in features:
            cols, part = feature_columns(data.columns, feature)
            feature_cols += cols
            self._features_list.append((cols, part))

        self.data = data[feature_cols]
        self._features_index_list = []
        for cols, part in self._features_list:
            index = [self.data.columns.get_loc(c) for c in cols]
            self._features_index_list.append((index, part))

        self._config['features_list'] = self._features_list
        LOGGER.info(self._features_list)

    def load(self):
        return self.data.values, self.labels

    @property
    def feature_indices(self):
        return self._features_index_list

    @property
    def configuration(self):
        return self._config
