"""Creates intermediate datasets containing unique features."""
import os.path

from logzero import logger
import pandas as pd

from hit_prediction_code import common
from hit_prediction_code.dataloaders import matcher

path_prefix = 'data/hit_song_prediction_ismir2020/interim'

for dataset in ['cleaned_matches', 'exact_matches', 'non_matches']:
    logger.info('Read msd bb mbid %s' % dataset)
    msd_bb_mbid = pd.read_csv(
        os.path.join(
            path_prefix,
            'msd_bb_mbid_' + dataset + '.csv',
        ),
        header=0,
        index_col=0,
    )

    for source, min_length, join_col, sample_id_col in [
        ('ab', 120, 'mbid', 'file_id'),
        ('essentia', 25, 'msd_id', 'msd_id'),
    ]:
        for feature_type in ['hl', 'll']:
            logger.info('Read %s %s %s features' % (
                source,
                dataset,
                feature_type,
            ))
            filename = 'msd_bb_mbid_' + dataset + '_' + source + '_'
            filename += feature_type + '_features'
            data = pd.read_parquet(
                os.path.join(
                    path_prefix,
                    filename + '.parquet',
                ))
            data = data.merge(
                msd_bb_mbid,
                on=[join_col],
            ).drop_duplicates(subset=[sample_id_col])
            logger.info('Select %s features from %s' % (feature_type, source))
            numeric_cols = matcher.get_numeric_columns(data)
            features = common.get_columns_matching_list(
                numeric_cols,
                common.all_no_year_list(),
            )
            data = matcher.select_feature_samples(
                data=data,
                features=features,
                min_length=min_length,
                max_length=600,
            )
            logger.info('Store %s %s feature dataset' % (source, feature_type))
            data.to_parquet(
                os.path.join(
                    path_prefix,
                    filename + '_unique.parquet',
                ))
