import multiprocessing as mp

import click
from fuzzywuzzy import fuzz
import numpy as np
import pandas as pd

from hit_prediction_code.dataloaders import millionsongdataset

RESULT_PATH = 'data/processed'
BB_PATH = '/storage/nas3/datasets/music/billboard'
MSD_PATH = 'data/millionsongdataset'


@click.group()
@click.option('--path',
              default='.',
              help='The path where the results are stored.')
def cli(path):
    global RESULT_PATH
    RESULT_PATH = path


def main1():
    msd_track_duplicates()


def main2():
    msd = millionsongdataset.read_msd_unique_tracks()
    year = millionsongdataset.read_msd_tracks_per_year()[['msd_id', 'year']]
    billboard = read_billboard_tracks()
    features = read_msd_feature_files()

    msd = join(msd, year, on=['msd_id'])
    msd = join(msd, features, on=['msd_id'])

    matches = join(msd, billboard, on=['artist', 'title'])

    duplicates = matches[matches.duplicated(subset=['artist', 'title'],
                                            keep=False)]
    duplicates.to_csv(RESULT_PATH + '/msd_bb_matches_duplicates.csv')

    results = join(msd, billboard, on=['artist', 'title'], how='left')

    duplicates = results[results.duplicated(subset=['artist', 'title'],
                                            keep=False)]
    duplicates.to_csv(RESULT_PATH + '/msd_bb_all_duplicates.csv')


@cli.command()
def match():
    msd = millionsongdataset.read_msd_unique_tracks()
    year = millionsongdataset.read_msd_tracks_per_year()[['msd_id', 'year']]
    billboard = read_billboard_tracks()
    features = read_msd_feature_files()

    msd = join(msd, year, on=['msd_id'])
    msd = join(msd, features, on=['msd_id'])

    matches = join(msd, billboard, on=['artist', 'title'])
    keep_first_duplicate(matches)
    matches.to_csv(RESULT_PATH + '/msd_bb_matches.csv')

    results = join(msd, billboard, on=['artist', 'title'], how='left')
    keep_first_duplicate(results)
    results.to_csv(RESULT_PATH + '/msd_bb_all.csv')

    df_split = np.array_split(results, mp.cpu_count() * 4)

    with mp.Pool() as pool:
        result_entries = pool.imap_unordered(_fuzzy_match, df_split)
        fuzzy_results = pd.DataFrame(columns=list(msd.columns) +
                                     ['max_sim', 'artist_sim', 'title_sim'])
        for result in result_entries:
            fuzzy_results = fuzzy_results.append(result,
                                                 ignore_index=True,
                                                 sort=False)
        fuzzy_results.to_csv(RESULT_PATH + '/msd_bb_fuzzy_matches.csv')

        fuzzy_results = fuzzy_results.loc[fuzzy_results['title_sim'] <= 40]
        fuzzy_results = fuzzy_results[[
            'msd_id',
            'echo_nest_id',
            'artist',
            'title',
            'year',
        ]]
        fuzzy_results.to_csv(RESULT_PATH + '/msd_bb_non_matches.csv')


def _fuzzy_match(msd):
    billboard = read_billboard_tracks()
    results = pd.DataFrame(columns=list(msd.columns) +
                           ['max_sim', 'artist_sim', 'title_sim'])
    for _, row_msd in msd.iterrows():
        entry = {
            **row_msd,
            'max_sim': 0,
        }
        for _, row_bb in billboard.iterrows():
            artist_sim, title_sim = fuzz.ratio(row_msd['artist'],
                                               row_bb['artist']), fuzz.ratio(
                                                   row_msd['title'],
                                                   row_bb['title'])
            sim = fuzz.ratio(row_msd['artist'] + '|#|' + row_msd['title'],
                             row_bb['artist'] + '|#|' + row_bb['title'])
            if sim > entry['max_sim']:
                entry['max_sim'] = sim
                entry['artist_sim'] = artist_sim
                entry['title_sim'] = title_sim
                entry['peak'] = row_bb['peak']
                entry['weeks'] = row_bb['weeks']
        entry = pd.Series(entry)
        results = results.append(entry, ignore_index=True)

    return results


def keep_first_duplicate(data):
    data.drop_duplicates(subset=['artist', 'title'],
                         keep='first',
                         inplace=True)


def remove_duplicates(data):
    data.drop_duplicates(subset=['artist', 'title'], keep=False, inplace=True)
    data.drop_duplicates(subset=['echo_nest_id'], keep=False, inplace=True)


def match_and_store_datasets(left,
                             right,
                             output_file,
                             how='inner',
                             hdf=None,
                             key='data'):
    combined = join(left, right, on=['artist', 'title'], how=how)
    if hdf:
        combined.to_hdf(output_file, key=key)
    else:
        combined.to_csv(output_file)


def join(left, right, on, how='inner'):
    return pd.merge(left, right, how=how, left_on=on, right_on=on)


def bb_track_duplicates():
    bb = read_billboard_tracks()
    tracks = bb.groupby(['artist', 'title'])

    for index, group in tracks:
        group_cnt = group.count()['peak']
        if group_cnt > 1:
            print(index, group_cnt)


def msd_track_duplicates():
    msd = millionsongdataset.read_msd_unique_tracks()
    unique_file_count = len(set(msd['msd_id']))
    unique_id_count = len(set(msd['echo_nest_id']))
    print(str(unique_file_count) + ',' + str(unique_id_count))

    tracks = msd.groupby(['artist', 'title'])

    count = 0
    for index, group in tracks:
        group_cnt = group.count()['msd_id']
        if group_cnt > 1:
            for item in group['msd_id']:
                output_line = item + ',' + index
                print(output_line)
            count += 1

    print(len(tracks), count)


def read_msd_feature_files():
    file_path = MSD_PATH + '/msd_audio_features/file_ids.csv'

    return pd.read_csv(file_path, header=None, names=['msd_id'])


def read_billboard_tracks():
    file_path = BB_PATH + '_mp3/billboard_1954-2018_summary.csv'

    return pd.read_csv(file_path)


if __name__ == '__main__':
    cli()
