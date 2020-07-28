"""Loads melspects for mbids in the msd_bb_mbid dataset."""
import os.path
import warnings

from logzero import logger

from hit_prediction_code.dataloaders import melspect

warnings.simplefilter('ignore')

path_prefix = 'data/hit_song_prediction_ismir2020/interim'

logger.info('Combine melspect features for msd_bb_mbid')
datasets = [
    'msd_bb_mbid_cleaned_matches',
    'msd_bb_mbid_exact_matches',
    'msd_bb_mbid_non_matches',
]
for filename in datasets:
    filename = os.path.join(path_prefix, filename + '.csv')
    melspect.combine_with_dataset(filename)
