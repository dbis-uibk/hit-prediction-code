"""Extracts year information for UUIDs."""
import os.path

from logzero import logger
import pandas as pd

interim_prefix = 'data/hit_song_prediction_ismir2020/interim'
final_prefix = 'data/hit_song_prediction_ismir2020/processed'
msd_prefix = 'data/millionsongdataset/additional_files'

logger.info('load uuid msd_id mapping')
matches = pd.read_csv(
    os.path.join(interim_prefix, 'msd_bb_mbid_cleaned_matches.csv'),
    header=0,
)[['uuid', 'msd_id']]
matches_exact = pd.read_csv(
    os.path.join(interim_prefix, 'msd_bb_mbid_exact_matches.csv'),
    header=0,
)[['uuid', 'msd_id']]
non_matches = pd.read_csv(
    os.path.join(interim_prefix, 'msd_bb_mbid_non_matches.csv'),
    header=0,
)[['uuid', 'msd_id']]
uuid_msd_mapping = pd.concat([
    matches,
    matches_exact,
    non_matches,
]).drop_duplicates()
assert len(uuid_msd_mapping) == len(
    uuid_msd_mapping['msd_id'].drop_duplicates()), 'msd_ids not unique'

logger.info('load msd year info')
msd_year = pd.read_csv(
    os.path.join(msd_prefix, 'tracks_per_year.txt'),
    sep='<SEP>',
    names=['year', 'msd_id', 'artist', 'title'],
)[['year', 'msd_id']].drop_duplicates()
assert len(msd_year) == len(
    msd_year['msd_id'].drop_duplicates()), 'msd_ids year mapping not unique'

logger.info('compute year mapping')
year_mapping = uuid_msd_mapping.merge(
    msd_year,
    on=['msd_id'],
)[['uuid', 'year']]

logger.info('get uuids with known year')
uuid_year = year_mapping.groupby(by='uuid').agg(lambda x: x.max() == x.min())
uuid_year = uuid_year[uuid_year['year']].reset_index()[['uuid']]
assert len(uuid_year) == len(
    uuid_year['uuid'].drop_duplicates()), 'UUIDs in year mapping not unique'

uuid_year_mapping = uuid_year.merge(year_mapping, on=['uuid'])
assert len(uuid_year_mapping) == len(uuid_year_mapping['uuid'].drop_duplicates(
)), 'UUIDs year mapping not unique'

uuid_year_mapping.to_csv(
    os.path.join(
        final_prefix,
        'msd_bb_mbid_uuid_year.csv',
    ))
