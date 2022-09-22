"""Merges mel-spect features to the final dataset."""
import glob
import os.path

from logzero import logger
import pandas as pd

path_prefix = 'data/hit_song_prediction_lastfm/interim'
final_prefix = 'data/hit_song_prediction_lastfm/processed'

dataset_prefix = 'msd_lastfm_matches_melspect_'

logger.info('Gather list of mel-spect files')
feature_files = glob.glob(
    os.path.join(
        path_prefix,
        dataset_prefix + 'features_*.pickle.xz',
    ))

logger.info('Load full dataset without features')
dataset = pd.read_pickle(
    os.path.join(
        final_prefix,
        dataset_prefix + 'unique.pickle.xz',
    ))
assert len(dataset) == len(set(dataset['uuid'])), 'Duplicates in dataset.'

columns = ['uuid', 'librosa_melspectrogram']

features = pd.DataFrame(columns=columns)
for feature_file in feature_files:
    logger.info(f'Load partial feature file: \'{feature_file}\'')
    features = pd.concat([features, pd.read_pickle(feature_file)[columns]])

assert len(features) == len(set(features['uuid'])), 'Duplicates in features.'

logger.info('Merge partial features with final dataset')
dataset = dataset.merge(features, on=['uuid'])

if len(dataset) != len(features):
    logger.warning('There are %d features for %d songs.' %
                   (len(features), len(dataset)))

logger.info('Store dataset with features containing %d songs' %
            len(dataset.index))
dataset.to_pickle(
    os.path.join(
        final_prefix,
        dataset_prefix + 'features_unique.pickle',
    ))
