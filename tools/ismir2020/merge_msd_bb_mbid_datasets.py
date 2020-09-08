"""Create datasets based on msd bb mbid."""
import os.path

from logzero import logger
import pandas as pd
from sklearn.utils import shuffle

path_prefix = 'data/hit_song_prediction_ismir2020/interim'
final_prefix = 'data/hit_song_prediction_ismir2020/processed'

logger.info('Compute common uuid per dataset')
all_uuid = {}
datasets = {}
for dataset in ['cleaned_matches', 'exact_matches', 'non_matches']:
    datasets[dataset] = {}
    for source in ['ab', 'essentia']:
        filename = 'msd_bb_mbid_' + dataset + '_' + source
        filename += '_unique_features.parquet'
        filename = os.path.join(path_prefix, filename)

        data = pd.read_parquet(filename)
        current_uuid = data[['uuid']]
        datasets[dataset][source] = data

        if all_uuid.get(dataset, None) is None:
            all_uuid[dataset] = current_uuid
        else:
            all_uuid[dataset] = all_uuid[dataset].merge(
                current_uuid,
                on=['uuid'],
            )

logger.info('Sample non hits datasets')
for dataset in ['cleaned_matches', 'exact_matches']:
    non_hit_sample = all_uuid['non_matches'].sample(
        all_uuid[dataset].shape[0],
        random_state=42,
    )
    all_uuid[dataset] = all_uuid[dataset].append(non_hit_sample)
del all_uuid['non_matches']

logger.info('Load msd bb mbid')
msd_bb_mbid_info = None
for dataset in ['cleaned_matches', 'exact_matches', 'non_matches']:
    logger.info('Read msd bb mbid %s' % dataset)
    current = pd.read_csv(
        os.path.join(
            path_prefix,
            'msd_bb_mbid_' + dataset + '.csv',
        ),
        header=0,
        index_col=0,
    )

    if msd_bb_mbid_info is None:
        msd_bb_mbid_info = current
    else:
        msd_bb_mbid_info = msd_bb_mbid_info.append(current)
msd_bb_mbid_info = msd_bb_mbid_info.drop_duplicates(['uuid'])

logger.info('Generate datasets')
targets = pd.read_csv(
    os.path.join(
        path_prefix,
        'msd_bb_mbid_targets.csv',
    ),
    header=0,
    index_col=0,
)

for dataset, uuids in all_uuid.items():
    for source, data in datasets[dataset].items():
        data = data.merge(uuids, on=['uuid'])
        non_hit_data = datasets['non_matches'][source].merge(
            uuids,
            on=['uuid'],
        )
        data = data.append(non_hit_data)

        filename = os.path.join(
            final_prefix,
            'msd_bb_mbid_' + dataset + '_' + source + '_unique.parquet',
        )
        logger.info('Store %s %s containing %d songs' %
                    (dataset, source, len(data.index)))
        data = data.merge(msd_bb_mbid_info, on=['uuid'])
        data = data.merge(targets, on=['uuid'])
        data = shuffle(data, random_state=42)
        data.to_parquet(filename)
