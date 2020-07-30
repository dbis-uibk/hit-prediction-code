"""loads acoustic brainz features for mbids in msd_bb_mbid dataset."""
import os.path

from logzero import logger
import pandas as pd

from hit_prediction_code.dataloaders import acousticbrainz

path_prefix = 'data/hit_song_prediction_ismir2020/interim'


def _load_features(name, feature):
    songs = pd.read_csv(os.path.join(path_prefix, name + '.csv'))
    songs.drop_duplicates(['mbid'])
    mbids = list(songs['mbid'])

    logger.info('Combine %s features for %s' % (feature, name))
    acousticbrainz.load_ab_features_as_df(mbids, feature).to_parquet(
        os.path.join(
            path_prefix,
            name + '_ab_' + feature + '_features.parquet',
        ))


# combine all features
logger.info('Combine acousticbrainz features for msd_bb_mbid')
datasets = [
    'msd_bb_mbid_cleaned_matches',
    'msd_bb_mbid_exact_matches',
    'msd_bb_mbid_non_matches',
]
for filename in datasets:
    for feature_type in ['ll', 'hl']:
        _load_features(filename, feature_type)
