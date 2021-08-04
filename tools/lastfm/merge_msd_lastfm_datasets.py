"""Create datasets based on msd lastfm."""
import os.path

from logzero import logger
import pandas as pd
from sklearn.utils import shuffle

path_prefix = 'data/hit_song_prediction_lastfm/interim'
final_prefix = 'data/hit_song_prediction_lastfm/processed'

logger.info('Compute common uuid per dataset')
all_uuid = None
dataset = {}
dataset_name = 'msd_lastfm_matches'
for source in ['ab', 'essentia', 'melspect']:
    filename = dataset_name + '_' + source
    if source == 'melspect':
        filename += '_features.pickle.xz'
    else:
        filename += '_unique_features.parquet'
    filename = os.path.join(path_prefix, filename)

    if source == 'melspect':
        data = pd.read_pickle(filename)
    else:
        data = pd.read_parquet(filename)
    current_uuid = data[['uuid']]
    dataset[source] = data

    if all_uuid is None:
        all_uuid = current_uuid
    else:
        all_uuid = all_uuid.merge(
            current_uuid,
            on=['uuid'],
        )

logger.info('Generate datasets')
targets = pd.read_csv(
    os.path.join(
        path_prefix,
        'msd_lastfm_matches_targets.csv',
    ),
    header=0,
    index_col=0,
)
targets = targets[['uuid', 'lastfm_listener_count', 'lastfm_playcount']]
targets.drop_duplicates(inplace=True)
assert len(targets) == len(set(targets['uuid'])), 'uuid not unique'

for source, data in dataset.items():
    data = data.merge(all_uuid, on=['uuid'])

    if source == 'melspect':
        file_suffix = '_unique.pickle.xz'
    else:
        file_suffix = '_unique.parquet'

    filename = os.path.join(
        final_prefix,
        dataset_name + '_' + source + file_suffix,
    )
    logger.info('Add targests to %s %s containing %d songs' %
                (dataset_name, source, len(data.index)))
    data = data.merge(targets, on=['uuid'], how='left')
    data = shuffle(data, random_state=42)
    logger.info('Store %s %s containing %d songs' %
                (dataset_name, source, len(data.index)))

    assert len(targets) == len(set(targets['uuid'])), 'uuid not unique'

    if source == 'melspect':
        data.to_pickle(filename, 'xz')
    else:
        data.to_parquet(filename)
