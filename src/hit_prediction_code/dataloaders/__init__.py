"""Dataloaders for the hit song prediction."""
import json
import logging
import os.path

from dbispipeline.base import Loader
import numpy as np
import pandas as pd
from sklearn import preprocessing

from ..common import feature_columns

LOGGER = logging.getLogger(__name__)


class MsdBbLoader(Loader):
    """Million song dataset / billboard charts loader.

    This loader is depricated please look at the EssentiaLoader.

    """

    def __init__(self,
                 hits_file_path,
                 non_hits_file_path,
                 features_path,
                 non_hits_per_hit=None,
                 features=None,
                 label=None,
                 nan_value=0,
                 random_state=None):
        """Initializes the msd billboard loader.

        Args:
            hits_file_path: the path to the csv containing all hit songs.
            non_hits_file_path: the path to the csv containing all non-hits.
            features_path: the path to the directory containing features in
                json format.
            non_hits_per_hit: the number of non-hits per hit in the resulting
                dataset.
            features: a list of selected columns used as features.
            label: the column selected as the target variable.
            nan_value: the values used for NaN values in the dataset.
            random_state: the random state used for non-hit subsampling.

        """
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
        ll_features = pd.read_hdf(
            os.path.join(
                features_path,
                'msd_bb_ll_features.h5',
            ))
        data = data.merge(ll_features, on='msd_id')
        hl_features = pd.read_hdf(
            os.path.join(
                features_path,
                'msd_bb_hl_features.h5',
            ))
        data = data.merge(hl_features, on='msd_id')
        data = _key_mapping(data)

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
        """Returns the data loaded by the dataloader."""
        return self.data.values, self.labels

    @property
    def feature_indices(self):
        """Returns the mapping between features and indeces."""
        return self._features_index_list

    @property
    def configuration(self):
        """Returns the configuration in json serializable format."""
        return self._config


def _get_highlevel_feature(features_path, msd_id):
    file_suffix = '.mp3.highlevel.json'
    return _load_feature(features_path, msd_id, file_suffix)


def _get_lowlevel_feature(features_path, msd_id):
    file_suffix = '.mp3'
    return _load_feature(features_path, msd_id, file_suffix)


def _load_feature(features_path, msd_id, file_suffix):
    file_prefix = 'features_tracks_' + msd_id[2].lower()
    file_name = os.path.join(
        features_path,
        file_prefix,
        msd_id + file_suffix,
    )

    with open(file_name) as features:
        return json.load(features)


def _key_mapping(df):
    # https://stackoverflow.com/questions/37292872/how-can-i-one-hot-encode-in-python
    string_columns = [
        'tonal.chords_key',
        'tonal.chords_scale',
        'tonal.key_scale',
        'tonal.key_key',
        'artist',
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

    def __init__(self,
                 dataset_path,
                 features='librosa_melspectrogram',
                 label=None,
                 nan_value=0,
                 binarize_labels=False):
        """Initializes the mel spect loader.

        Args:
            dataset_path: the path to the pickeled dataset.
            features: a column or a list of selected columns used as features.
            label: the column selected as the target variable.
            nan_value: the values used for NaN values in the dataset.
            binarize_labels: specifies if sklearns LabelBinarizer is applied to
                the target values.

        """
        self._config = {
            'dataset_path': dataset_path,
            'features': features,
            'label': label,
            'nan_value': nan_value,
            'binarize_labels': binarize_labels,
        }

        data = pd.read_pickle(dataset_path)

        self.labels = np.ravel(data[[label]])
        nan_values = pd.isnull(self.labels)
        self.labels[nan_values] = nan_value

        # ensure that the array is at least 2d
        if len(self.labels.shape) == 1:
            self.labels = self.labels.reshape((*self.labels.shape, 1))

        if self._config['binarize_labels']:
            label_binarizer = preprocessing.LabelBinarizer()
            self.labels = label_binarizer.fit_transform(self.labels)

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

    def __init__(self,
                 dataset_path,
                 features,
                 label=None,
                 nan_value=0,
                 label_modifier=None,
                 binarize_labels=False):
        """Initializes the essentia loader.

        Args:
            dataset_path: the path to the pickeled dataset.
            features: a list of selected columns used as features.
            label: the column selected as the target variable.
            nan_value: the values used for NaN values in the dataset.
            label_modifier: function applied to each label.
            binarize_labels: specifies if sklearns LabelBinarizer is applied to
                the target values.
        """
        self._config = {
            'dataset_path': dataset_path,
            'features': features,
            'label': label,
            'nan_value': nan_value,
            'binarize_labels': binarize_labels,
        }

        data = pd.read_pickle(dataset_path)
        data = _key_mapping(data)

        self.labels = np.ravel(data[[label]])
        nan_values = pd.isnull(self.labels)
        self.labels[nan_values] = nan_value

        if label_modifier is not None:
            self.labels = label_modifier(self.labels)

        if self._config['binarize_labels']:
            label_binarizer = preprocessing.LabelBinarizer()
            self.labels = label_binarizer.fit_transform(self.labels)

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
        """Returns the data loaded by the dataloader."""
        return self.data.values, self.labels

    @property
    def feature_indices(self):
        """Returns the mapping between features and indeces."""
        return self._features_index_list

    @property
    def configuration(self):
        """Returns the configuration in json serializable format."""
        return self._config
