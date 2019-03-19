import csv
import sys

import pandas as pd


def main1():
    msd_track_duplicates()


def main():
    msd = read_msd_tracks_per_year()
    echo = read_msd_unique_tracks()[['msd_id', 'echo_nest_id']]
    bb = read_billboard_tracks()
    features = read_msd_feature_files()

    msd = join(msd, features, on=['msd_id'])
    msd = join(msd, echo, on=['msd_id'])

    msd.drop_duplicates(subset=['artist', 'title'], keep=False, inplace=True)
    msd.drop_duplicates(subset=['echo_nest_id'], keep=False, inplace=True)

    match_and_store_datasets(msd, bb, 'msd_bb_matches.csv')
    match_and_store_datasets(msd, bb, 'msd_bb_all.csv', how='left')


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


if __name__ == '__main__':
    main()
