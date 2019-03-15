import csv
import sys

import pandas as pd


def main():
    match_msd_billboard()


def match_msd_billboard():
    msd = read_msd_tracks()
    billboard = read_billboard_tracks()

    combined = pd.merge(
        msd,
        billboard,
        how='inner',
        left_on=['artist', 'title'],
        right_on=['artist', 'title'])
    with open('msd_bb_matches.csv', 'w') as out_file:
        combined.to_csv(out_file)


def msd_track_duplicates():
    msd = read_msd_tracks()
    unique_file_count = len(set(msd['track_file']))
    unique_id_count = len(set(msd['track_id']))
    print(str(unique_file_count) + ',' + str(unique_id_count))

    tracks = msd.groupby(['track_id'])

    for index, group in tracks:
        group_cnt = group.count()['track_file']
        if group_cnt > 1:
            for item in group['track_file']:
                output_line = item + ',' + index
                print(output_line)


def read_msd_tracks():
    file_path = '/storage/nas3/datasets/music/millionsongdataset/track_titles_artists.csv'

    return pd.read_csv(
        file_path,
        sep='<SEP>',
        header=None,
        names=['track_file', 'track_id', 'artist', 'title'])


def read_billboard_tracks():
    file_path = '/storage/nas3/datasets/music/billboard_mp3/billboard_1954-2018_summary.csv'

    return pd.read_csv(file_path)


if __name__ == '__main__':
    main()
