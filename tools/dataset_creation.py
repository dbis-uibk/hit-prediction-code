import csv
import json
import sys
import multiprocessing as mp

import click

from fuzzywuzzy import fuzz
from fuzzywuzzy import process

import numpy as np

import pandas as pd


@click.group()
def cli():
    pass


def main1():
    msd_track_duplicates()


def main2():
    msd = read_msd_unique_tracks()
    year = read_msd_tracks_per_year()[['msd_id', 'year']]
    billboard = read_billboard_tracks()
    features = read_msd_feature_files()

    msd = join(msd, year, on=['msd_id'])
    msd = join(msd, features, on=['msd_id'])

    matches = join(msd, billboard, on=['artist', 'title'])

    duplicates = matches[matches.duplicated(
        subset=['artist', 'title'], keep=False)]
    duplicates.to_csv('msd_bb_matches_duplicates.csv')

    results = join(msd, billboard, on=['artist', 'title'], how='left')

    duplicates = results[results.duplicated(
        subset=['artist', 'title'], keep=False)]
    duplicates.to_csv('msd_bb_all_duplicates.csv')


@cli.command()
def match():
    msd = read_msd_unique_tracks()
    year = read_msd_tracks_per_year()[['msd_id', 'year']]
    billboard = read_billboard_tracks()
    features = read_msd_feature_files()

    msd = join(msd, year, on=['msd_id'])
    msd = join(msd, features, on=['msd_id'])

    matches = join(msd, billboard, on=['artist', 'title'])
    keep_first_duplicate(matches)
    matches.to_csv('msd_bb_matches.csv')

    results = join(msd, billboard, on=['artist', 'title'], how='left')
    keep_first_duplicate(results)
    results.to_csv('msd_bb_all.csv')

    df_split = np.array_split(results, mp.cpu_count() * 4)

    with mp.Pool() as pool:
        result_entries = pool.imap_unordered(work, df_split)
        fuzzy_results = pd.DataFrame(
            columns=list(msd.columns) + ['max_sim', 'artist_sim', 'title_sim'])
        for result in result_entries:
            fuzzy_results = fuzzy_results.append(
                result, ignore_index=True, sort=False)
        fuzzy_results.to_csv('msd_bb_fuzzy_matches.csv')

        fuzzy_results = fuzzy_results.loc[fuzzy_results['title_sim'] <= 40]
        fuzzy_results = fuzzy_results[[
            'msd_id', 'echo_nest_id', 'artist', 'title', 'year'
        ]]
        fuzzy_results.to_csv('msd_bb_non_matches.csv')


@cli.command()
def combine_lowlevel_features():
    hits = set(read_hits()['msd_id'])
    non_hits = set(read_non_hits()['msd_id'])
    msd_ids = hits + non_hits
    features = _combine_ll_features(msd_ids)
    features.to_hdf('msd_bb_ll_features.h5', 'll')


def _combine_ll_features(msd_ids):
    features_path = '/storage/nas3/datasets/music/millionsongdataset/msd_audio_features'  # noqa E501

    ll_features = pd.DataFrame()
    for msd_id in msd_ids:
        file_id = pd.DataFrame([msd_id], columns=['msd_id'])
        feature = pd.io.json.json_normalize(
            _get_lowlevel_feature(features_path, msd_id))
        ll_features = ll_features.append(
            file_id.join(feature), sort=False, ignore_index=True)

    return ll_features


@cli.command()
def combine_highlevel_features():
    hits = set(read_hits()['msd_id'])
    non_hits = set(read_non_hits()['msd_id'])
    msd_ids = hits + non_hits
    features = _combine_hl_features(msd_ids)
    features.to_hdf('msd_bb_hl_features.h5', 'hl')


def _combine_hl_features(msd_ids):
    features_path = '/storage/nas3/datasets/music/millionsongdataset/msd_audio_features'  # noqa E501

    hl_features = pd.DataFrame()
    for msd_id in msd_ids:
        file_id = pd.DataFrame([msd_id], columns=['msd_id'])
        feature = pd.io.json.json_normalize(
            _get_highlevel_feature(features_path, msd_id))
        hl_features = hl_features.append(
            file_id.join(feature), sort=False, ignore_index=True)

    return hl_features


def work(msd):
    billboard = read_billboard_tracks()
    results = pd.DataFrame(
        columns=list(msd.columns) + ['max_sim', 'artist_sim', 'title_sim'])
    for _, row_msd in msd.iterrows():
        entry = {
            **row_msd,
            'max_sim': 0,
        }
        for _, row_bb in billboard.iterrows():
            artist_sim, title_sim = fuzz.ratio(
                row_msd['artist'], row_bb['artist']), fuzz.ratio(
                    row_msd['title'], row_bb['title'])
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
    data.drop_duplicates(
        subset=['artist', 'title'], keep='first', inplace=True)


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
    msd = read_msd_unique_tracks()
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


def _get_highlevel_feature(features_path, msd_id):
    file_suffix = '.mp3.highlevel.json'
    return _load_feature(features_path, msd_id, file_suffix)


def _get_lowlevel_feature(features_path, msd_id):
    file_suffix = '.mp3'
    return _load_feature(features_path, msd_id, file_suffix)


def _load_feature(features_path, msd_id, file_suffix):
    file_prefix = '/features_tracks_' + msd_id[2].lower() + '/'
    file_name = features_path + file_prefix + msd_id + file_suffix

    with open(file_name) as features:
        return json.load(features)


def read_msd_tracks_per_year():
    file_path = '/storage/nas3/datasets/music/millionsongdataset/additional_files/tracks_per_year.txt'  # noqa E501

    return pd.read_csv(
        file_path,
        sep='<SEP>',
        header=None,
        names=['year', 'msd_id', 'artist', 'title'])


def read_msd_unique_artists():
    file_path = '/storage/nas3/datasets/music/millionsongdataset/additional_files/unique_tracks.txt'  # noqa E501

    return pd.read_csv(
        file_path,
        sep='<SEP>',
        header=None,
        names=['artist_id', 'mb_artist_id', 'msd_id', 'artist'])


def read_msd_unique_tracks():
    file_path = '/storage/nas3/datasets/music/millionsongdataset/additional_files/unique_tracks.txt'  # noqa E501

    return pd.read_csv(
        file_path,
        sep='<SEP>',
        header=None,
        names=['msd_id', 'echo_nest_id', 'artist', 'title'])


def read_msd_feature_files():
    file_path = '/storage/nas3/datasets/music/millionsongdataset/msd_audio_features/file_ids.csv'  # noqa E501

    return pd.read_csv(file_path, header=None, names=['msd_id'])


def read_billboard_tracks():
    file_path = '/storage/nas3/datasets/music/billboard_mp3/billboard_1954-2018_summary.csv'  # noqa E501

    return pd.read_csv(file_path)


def read_hits():
    file_path = '/storage/nas3/datasets/music/billboard/msd_bb_matches.csv'
    return pd.read_csv(file_path)


def read_non_hits():
    file_path = '/storage/nas3/datasets/music/billboard/msd_bb_non_matches.csv'
    return pd.read_csv(file_path)


if __name__ == '__main__':
    cli()
