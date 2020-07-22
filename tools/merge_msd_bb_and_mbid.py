"""Merges the ismir2019 dataset with mbids."""
from logzero import logger
import pandas as pd

logger.info('Merge msd_bb_matches with mbid')
msd_mbid_map = pd.read_csv(
    'data/hit_song_prediction_ismir2020/raw/msd-mbid-2016-01-results-ab.csv',
    names=['msd_id', 'mbid', 'title', 'artist'],
)

logger.info('Load msd_bb_matches.csv')
matches = pd.read_csv(
    'data/hit_song_prediction_ismir2019/msd_bb_matches.csv',
    index_col=0,
)

matches_mbid = matches.merge(
    msd_mbid_map,
    on=['msd_id'],
    suffixes=('_msd_bb', '_msd_mbid'),
)

logger.info('Store msd_bb_mbid_matches.csv')
matches_mbid.to_csv(
    'data/hit_song_prediction_ismir2020/interim/msd_bb_mbid_matches.csv')

logger.info('Load msd_bb_non_matches.csv')
non_matches = pd.read_csv(
    'data/hit_song_prediction_ismir2019/msd_bb_non_matches.csv',
    index_col=0,
)
non_matches_mbid = non_matches.merge(
    msd_mbid_map,
    on=['msd_id'],
    suffixes=('_msd_bb', '_msd_mbid'),
)
logger.info('Store msd_bb_mbid_non_matches.csv')
non_matches_mbid.to_csv(
    'data/hit_song_prediction_ismir2020/interim/msd_bb_mbid_non_matches.csv')
