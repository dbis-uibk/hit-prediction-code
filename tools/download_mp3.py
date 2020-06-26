"""Script for downloading mp3s."""
import pandas as pd
import youtube_dl

TARGET_DIRECTORY = 'data/interim/lfm_popularity/mp3s'


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
        except Exception:
            pass


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
    for _, row in data.itterrows():
        download_yt_mp3_for_track(
            target_directory=target_directory,
            track_id=row[id_column],
            artist=row[artist_column],
            title=row[title_column],
        )


if __name__ == '__main__':
    DATA = pd.read_parquet('data/raw/lfm_popularity/dataset_20.parquet')
    download_mp3s(
        data=DATA,
        id_column='recording_mbid',
        artist_column='artist_name',
        title_column='recording_name',
    )
