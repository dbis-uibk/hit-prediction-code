"""loads acoustic brainz features for mbids in msd_bb_mbid dataset."""
import os.path

from logzero import logger
import pandas as pd

from hit_prediction_code.dataloaders import acousticbrainz

path_prefix = 'data/hit_song_prediction_ismir2020/interim'


def _load_features(name, feature):
    songs = pd.read_csv(os.path.join(path_prefix, name + '.csv'))['mbid']
    mbids = set(songs)

    logger.info('Combine %s features for %s' % (feature, name))
    data = acousticbrainz.load_ab_features_as_df(mbids, feature)

    if feature == 'll':
        version_col = 'metadata.version.essentia_git_sha'
    else:
        version_col = 'metadata.version.lowlevel.essentia_git_sha'

    def is_clean_version(c):
        try:
            return not c.startswith('v2.1_beta1')
        except Exception as ex:
            logger.exception(ex)
            logger.error('c is: %s' % str(c))
            return True

    data = data[data[version_col].apply(is_clean_version)]

    data.to_parquet(
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
