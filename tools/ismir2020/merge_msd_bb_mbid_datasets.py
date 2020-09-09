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
    for source in ['ab', 'essentia', 'melspect']:
        filename = 'msd_bb_mbid_' + dataset + '_' + source
        if source == 'melspect':
            filename += '_features.pickle.xz'
        else:
            filename += '_unique_features.parquet'
        filename = os.path.join(path_prefix, filename)

        if source == 'melspect':
            data = pd.read_pickle(filename, 'xz')
        else:
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

        if source == 'melspect':
            file_suffix = '_unique.pickle.xz'
        else:
            file_suffix = '_unique.parquet'

        filename = os.path.join(
            final_prefix,
            'msd_bb_mbid_' + dataset + '_' + source + file_suffix,
        )
        logger.info('Store %s %s containing %d songs' %
                    (dataset, source, len(data.index)))
        data = data.merge(targets, on=['uuid'])
        data = shuffle(data, random_state=42)
        if source == 'melspect':
            data.to_pickle(filename, 'xz')
        else:
            data.to_parquet(filename)
