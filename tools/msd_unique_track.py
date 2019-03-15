import csv
import sys

import pandas as pd


def main():
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
        names=['track_file', 'track_id', 'track_artist', 'track_title'])


if __name__ == '__main__':
    main()
