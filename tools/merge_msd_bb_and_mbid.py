"""Merges the ismir2019 dataset with mbids."""
from logzero import logger
import pandas as pd

from hit_prediction_code.dataloaders import matcher

data_path = 'data/hit_song_prediction_ismir2020'

logger.info('Merge msd_bb_matches with mbid')

logger.info('Load msd mbid map')
msd_mbid_map = pd.read_csv(
    data_path + '/raw/msd-mbid-2016-01-results-ab.csv',
    names=['msd_id', 'mbid', 'title', 'artist'],
)[[
    'msd_id',
    'mbid',
]]
msd_mbid_map.drop_duplicates(inplace=True)

logger.info('Load lastfm info')
lastfm_info = pd.read_pickle(data_path + '/interim/msd_mbid_lastfm.pickle')[[
    'mbid',
    'lastfm_mbid',
]]
lastfm_info.drop_duplicates(inplace=True)

msd_mbid_map = msd_mbid_map.merge(lastfm_info, on=['mbid'])

logger.info('Load msd_bb_cleaned_matches.csv')
matches = pd.read_csv(
    data_path + '/interim/msd_bb_cleaned_matches.csv',
    header=0,
    index_col=0,
)[[
    'msd_id',
    'echo_nest_id',
    'title_clean',
    'artist_clean',
]]
matches.drop_duplicates(inplace=True)

matches_mbid = matches.merge(
    msd_mbid_map,
    on=['msd_id'],
)

logger.info('Assign uuid for each song')
matches_mbid = matcher.add_uuid_column(data=matches_mbid)

logger.info('Store msd_bb_mbid_cleaned_matches.csv')
matches_mbid.to_csv(data_path + '/interim/msd_bb_mbid_cleaned_matches.csv')

logger.info('Load msd_bb_exact_matches.csv')
matches = pd.read_csv(
    data_path + '/interim/msd_bb_exact_matches.csv',
    header=0,
    index_col=0,
)[[
    'msd_id',
    'echo_nest_id',
    'title',
    'artist',
]]
matches = matcher.clean_artist_title(data=matches)
del matches['title']
del matches['artist']
matches.drop_duplicates(inplace=True)

matches_mbid = matches.merge(
    msd_mbid_map,
    on=['msd_id'],
)

logger.info('Assign uuid for each song')
matches_mbid = matcher.add_uuid_column(data=matches_mbid)

logger.info('Store msd_bb_mbid_exact_matches.csv')
matches_mbid.to_csv(data_path + '/interim/msd_bb_mbid_exact_matches.csv')

logger.info('Load msd_bb_non_matches.csv')
non_matches = pd.read_csv(
    'data/hit_song_prediction_ismir2019/msd_bb_non_matches.csv',
    index_col=0,
)[[
    'msd_id',
    'echo_nest_id',
    'title',
    'artist',
]]
non_matches = matcher.clean_artist_title(data=non_matches)
del non_matches['title']
del non_matches['artist']
non_matches.drop_duplicates(inplace=True)

non_matches_mbid = non_matches.merge(
    msd_mbid_map,
    on=['msd_id'],
)

logger.info('Assign uuid for each song')
non_matches_mbid = matcher.add_uuid_column(data=non_matches_mbid)

logger.info('Store msd_bb_mbid_non_matches.csv')
non_matches_mbid.to_csv(data_path + '/interim/msd_bb_mbid_non_matches.csv')
