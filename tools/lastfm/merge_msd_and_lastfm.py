"""Merges the lastfm data with the million song dataset."""
from logzero import logger
import pandas as pd

from hit_prediction_code.dataloaders import matcher
from hit_prediction_code.dataloaders import millionsongdataset

data_path = 'data/hit_song_prediction_lastfm'
# TODO: Move files to raw as soon as the ismir2020 dataset gets released.
ismir_data_path = 'data/hit_song_prediction_ismir2020'

logger.info('Merge msd with lastfm')

logger.info('Load million song dataset')
msd = millionsongdataset.read_msd_unique_tracks()
msd = matcher.clean_artist_title(msd)[[
    'msd_id',
    'echo_nest_id',
    'title_clean',
    'artist_clean',
]]
msd.drop_duplicates(inplace=True)

logger.info('Load msd mbid map')
msd_mbid_map = pd.read_csv(
    ismir_data_path + '/raw/msd-mbid-2016-01-results-ab.csv',
    names=['msd_id', 'mbid', 'title', 'artist'],
)[[
    'msd_id',
    'mbid',
]]
msd_mbid_map.drop_duplicates(inplace=True)

logger.info('Load lastfm info')
lastfm_info = pd.read_pickle(ismir_data_path +
                             '/interim/msd_mbid_lastfm.pickle')[[
                                 'mbid',
                                 'lastfm_mbid',
                             ]]
lastfm_info.drop_duplicates(inplace=True)

msd_mbid_map = msd_mbid_map.merge(lastfm_info, on=['mbid'])

matches_mbid = msd.merge(
    msd_mbid_map,
    on=['msd_id'],
)

logger.info('Assign uuid for each song')
matches_mbid = matcher.add_uuid_column(data=matches_mbid)

logger.info('Store msd_lastfm_mbid_cleaned_matches.csv')
matches_mbid.to_csv(data_path + '/interim/msd_lastfm_matches.csv')
