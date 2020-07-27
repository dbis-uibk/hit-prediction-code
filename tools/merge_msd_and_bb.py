"""Merges the ismir2019 dataset with mbids."""
from logzero import logger

from hit_prediction_code.dataloaders import billboard
from hit_prediction_code.dataloaders import matcher
from hit_prediction_code.dataloaders import millionsongdataset

logger.info('Merge msd with bb')

logger.info('Load billboard hot 100')
bb = billboard.read_billboard_hot_100()

logger.info('Load million song dataset')
msd = millionsongdataset.read_msd_unique_tracks()

logger.info('Match exact msd and bb')
msd_bb = matcher.match_exact_msd_bb(msd, bb)
logger.info('Store exact matches')
msd_bb.to_csv(
    'data/hit_song_prediction_ismir2020/interim/msd_bb_exact_matches.csv')

logger.info('Match cleaned msd and bb')
msd_bb_clean = matcher.match_clean_msd_bb(msd, bb)
logger.info('Store cleaned matches')
msd_bb_clean.to_csv(
    'data/hit_song_prediction_ismir2020/interim/msd_bb_cleaned_matches.csv')
