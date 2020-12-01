"""Loads melspects for mbids in the msd_lastfm dataset."""
import os.path
import warnings

from logzero import logger
import pandas as pd

from hit_prediction_code.dataloaders import melspect

warnings.simplefilter('ignore')

path_prefix = 'data/hit_song_prediction_lastfm/interim'

logger.info('Combine melspect features for msd_lastfm')
dataset_name = 'msd_lastfm_matches'
filename = os.path.join(
    path_prefix,
    dataset_name + '_essentia_unique_features.parquet',
)
dataset = pd.read_parquet(filename)[['uuid', 'msd_id']]
dataset.drop_duplicates(inplace=True)
melspect.combine_with_dataset(
    dataset,
    filename,
    output_file_prefix=dataset_name,
    compression=None,
)
