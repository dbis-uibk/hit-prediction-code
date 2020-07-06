"""Script for downloading mp3s."""
from glob import glob
from random import random
from time import sleep
from urllib.error import HTTPError

import pandas as pd
import youtube_dl
from youtube_dl.utils import DownloadError

TARGET_DIRECTORY = 'data/interim/lfm_popularity/mp3s'
VISITED_FILE = TARGET_DIRECTORY + '/visited.csv'


def get_already_known_files():
    """Returns a list of already downloaded files."""
    mbid_start_idx = len(TARGET_DIRECTORY) + 1
    mbid_end_idx = mbid_start_idx + 36

    def mbid_extractor(name):
        return name[mbid_start_idx:mbid_end_idx]

    known_files = set(map(mbid_extractor, glob(TARGET_DIRECTORY + '/*.json')))
    known_files |= set(map(mbid_extractor, glob(TARGET_DIRECTORY + '/*.mp3')))

    return known_files


def get_visited_tracks():
    """Returns a set of visited tracks."""
    try:
        visited = pd.read_csv(VISITED_FILE)
        return set(visited['track_id'])
    except FileNotFoundError:
        return set()


def download_yt_mp3_for_track(target_directory, track_id, artist, title):
    """Downloads an mp3 file from youtube.

    Args:
        target_directory: dir to store the files.
        track_id: unique id for the track.
        artist: the name of the artist of the track.
        title: the title of the track.
    """
    dl_options = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        # "default_search": "ytsearch3",
        'outtmpl': f'{target_directory}/{track_id}_%(id)s.%(ext)s',
        'writeinfojson': True,
        'noplaylist': True,
    }
    with youtube_dl.YoutubeDL(dl_options) as ydl:
        # Ignore errors
        try:
            ydl.download([f'ytsearch1:{artist} {title}'])
        except DownloadError as ex:
            exc_class, exc, _ = ex.exc_info
            if exc_class == HTTPError and exc.code == 429:
                print(exc)
                exit()
        except Exception:
            pass

    sleep_time = 10. + random() * 50.
    print('sleep for', sleep_time, 'secs')
    sleep(sleep_time)


def download_mp3s(data,
                  id_column,
                  artist_column,
                  title_column,
                  target_directory=TARGET_DIRECTORY):
    """Download all files contained in a dataframe.

    Args:
        data: dataframe containing the songs to download files.
        id_column: column containing the unique ids.
        artist_column: column containing the artist name.
        title_column: column containing the title of the track.
        target_directory: directory to store the files.
    """
    count = 0
    random_count = int(4000. + random() * 2000.)

    visited_cols = ['track_id', 'artist', 'title']
    try:
        visited = pd.read_csv(VISITED_FILE, usecols=visited_cols)
    except FileNotFoundError:
        visited = pd.DataFrame(columns=visited_cols)

    for _, row in data.iterrows():
        visited = visited.append(
            {
                'track_id': row[id_column],
                'artist': row[artist_column],
                'title': row[title_column],
            },
            ignore_index=True,
        )
        visited.to_csv(VISITED_FILE)

        download_yt_mp3_for_track(
            target_directory=target_directory,
            track_id=row[id_column],
            artist=row[artist_column],
            title=row[title_column],
        )

        count += 1
        if count >= random_count:
            count = 0
            random_count = int(4000. + random() * 2000.)
            sleep_time = 1. + random() * 4.
            print('long sleep for', sleep_time, 'hrs')
            sleep(sleep_time * 3600)


def download_dataset():
    """Downloads all missing filse from the dataset."""
    id_column = 'recording_mbid'

    data = pd.read_parquet('data/raw/lfm_popularity/dataset_20.parquet')

    visited_tracks = get_visited_tracks()
    data = data[~data[id_column].isin(visited_tracks)]

    known_files = get_already_known_files()
    data = data[~data[id_column].isin(known_files)]

    download_mp3s(
        data=data,
        id_column=id_column,
        artist_column='artist_name',
        title_column='recording_name',
    )


if __name__ == '__main__':
    download_dataset()
