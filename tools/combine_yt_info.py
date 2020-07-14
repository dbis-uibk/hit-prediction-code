"""Script for downloading mp3s."""
from glob import glob
import json

import pandas as pd

TARGET_DIRECTORY = 'data/interim/lfm_popularity/mp3s'


def get_already_known_files():
    """Returns a list of already downloaded files."""
    mbid_start_idx = len(TARGET_DIRECTORY) + 1
    mbid_end_idx = mbid_start_idx + 36

    def mbid_extractor(name):
        return {'mbid': name[mbid_start_idx:mbid_end_idx], 'file': name}

    json_files = pd.DataFrame(
        map(mbid_extractor, glob(TARGET_DIRECTORY + '/*.json')))
    json_files.rename(columns={'file': 'info_json_file'}, inplace=True)

    mp3_files = pd.DataFrame(
        map(mbid_extractor, glob(TARGET_DIRECTORY + '/*.mp3')))
    mp3_files.rename(columns={'file': 'mp3_file'}, inplace=True)

    known_files = mp3_files.merge(json_files, on=['mbid'])

    return known_files


def combine_yt_info():
    """Combines all youtube info json files."""
    combined_info = get_already_known_files()

    yt_info = []
    for _, row in combined_info.iterrows():
        with open(row['info_json_file']) as json_file:
            info = json.load(json_file)
            info['mbid'] = row['mbid']
            yt_info.append(info)

    yt_info = pd.DataFrame(yt_info)
    yt_info.rename(
        lambda c: 'yt_' + c if not c == 'mbid' else c,
        axis='columns',
        inplace=True,
    )

    combined_info = combined_info.merge(yt_info, on=['mbid'])
    combined_info.to_pickle(TARGET_DIRECTORY + '/yt_info.pickle')


if __name__ == '__main__':
    combine_yt_info()
