"""loads acoustic brainz features for mbids in msd_bb_mbid dataset."""
import os.path

import pandas as pd

from hit_prediction_code.dataloaders import acousticbrainz

path_prefix = 'data/hit_song_prediction_ismir2020/interim'


def _load_features(name, feature):
    songs = pd.read_csv(os.path.join(path_prefix, name + '.csv'))
    mbids = list(songs['mbid'])

    acousticbrainz.load_ab_features_as_df(mbids, 'll').to_parquet(
        os.path.join(
            path_prefix,
            name + '_ab_' + feature + '_features.parquet',
        ))


# combine all features
for filename in ['msd_bb_mbid_matches', 'msd_bb_mbid_non_matches']:
    for feature_type in ['ll', 'hl']:
        _load_features(filename, feature_type)
