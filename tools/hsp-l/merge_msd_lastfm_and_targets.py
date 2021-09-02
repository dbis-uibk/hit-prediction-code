"""Merges dataset with target labels."""
import os

from logzero import logger
import pandas as pd

from hit_prediction_code.dataloaders import matcher

path_prefix = 'data/hit_song_prediction_lastfm/interim/'
# TODO: Move files to raw as soon as the ismir2020 dataset gets released.
ismir2020_path_prefix = 'data/hit_song_prediction_ismir2020/interim/'


def target_filter(data, target_col, keep_lowest):
    """Filters duplicates."""
    return matcher.filter_duplicates(
        data=data,
        id_cols=['uuid'],
        target_col=target_col,
        keep_lowest=keep_lowest,
    )


logger.info('Read msd lastfm info')
data = pd.read_csv(
    os.path.join(
        path_prefix,
        'msd_lastfm_matches.csv',
    ),
    header=0,
    index_col=0,
)
data.drop_duplicates(inplace=True)

logger.info('Read lastfm info')
lastfm = pd.read_pickle(ismir2020_path_prefix + 'msd_mbid_lastfm.pickle')[[
    'mbid',
    'lastfm_mbid',
    'lastfm_listener_count',
    'lastfm_playcount',
]]
lastfm = data.merge(lastfm, on=['mbid'])
lc = target_filter(
    lastfm[['uuid', 'lastfm_listener_count']],
    target_col='lastfm_listener_count',
    keep_lowest=False,
).drop_duplicates()
pc = target_filter(
    lastfm[['uuid', 'lastfm_playcount']],
    target_col='lastfm_playcount',
    keep_lowest=False,
).drop_duplicates()
lastfm_data = lc.merge(pc, on=['uuid'])

data = lastfm_data.merge(data, on=['uuid'])
data.drop_duplicates(inplace=True)

logger.info('Store targets')
data.to_csv(path_prefix + 'msd_lastfm_matches_targets.csv')
