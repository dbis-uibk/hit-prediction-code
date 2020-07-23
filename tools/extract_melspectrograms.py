#!/usr/bin/env python
"""Command line tool to extract melspectrogram features for a given dataset."""
import functools
import glob
import logging
import multiprocessing
import os
import sys
import tempfile
import warnings
import zipfile

import click
import librosa
import logzero
from logzero import logger
import pandas as pd

DATAFRAME_COLUMNS = [
    'archive_file_name',
    'msd_id',
    'librosa_melspectrogram',
]

INTERIM_PATH = os.path.join(
    'data',
    'hit_song_prediction_ismir2020',
    'interim',
)
PROCESSED_PATH = os.path.join(
    'data',
    'hit_song_prediction_ismir2020',
    'interim',
)
TMP_PATH = tempfile.gettempdir()

MP3_ARCHIVE_PATH = os.path.join(
    'data',
    'millionsongdataset',
    'mp3s',
)
OUTPUT_PREFIX = 'msd_librosa_melspect'


def extract_mel_from_mp3s_in_zipfile(zipfile_name,
                                     for_tracks=None,
                                     min_time=None,
                                     n_mels=128,
                                     window=None,
                                     window_size=None):
    """Extracts melspectrograms from mp3 files that are packed in a zip file.

    Args:
        zipfile_name: the path to the zip file containing the mp3s.
        for_tracks: a list containing the tracks specified by
            millionsongdataset-id (msd_id) for those we want the features. If
            None, all tracks are extracted.
        min_time: the minimum amout of time steps required for a audio sample.
        n_mels: the number of mel bands to generate.
        window: a function that takes the mel spectrogram as extracted by
            librosa as its first argument and the min_time which in this case
            is also the window size as its second argument. Also give a
            window_size if window is not None. Otherwise, a ValueError is
            raised.
        window_size: the size of the window that is used for subsampling. If
            the min_time is not None it needs to be at least as big as the
            window_size. Otherwise, it is set to window_size. If window_size
            needs to be set if window is not None.

    Returns:
        A pandas dataframe containing the name of the zip file, the name of the
        mp3 file and the extracted melspectrogram features for that file.

    """
    if min_time is None and window_size is not None:
        min_time = window_size

    if window is not None and window_size is None:
        raise ValueError('Pass a window_size if window is specified.')

    if window_size is not None:
        if window_size < 1:
            raise ValueError('The window_size needs to be at least 1')
        if min_time < window_size:
            raise ValueError('The window_size needs to be <= min_time.')
        if window is None:
            raise ValueError('Pass a window function if window_size is set.')

    result = []

    with zipfile.ZipFile(zipfile_name, 'r') as zip_object:
        archive_file_name = os.path.basename(zipfile_name)

        list_of_files = zip_object.namelist()

        # create a dataframe containing the path in the zip and the msd_id
        list_of_files = pd.DataFrame(
            zip(list_of_files, list_of_files),
            columns=['file_name', 'msd_id'],
        )

        # convert the path to the msd_id
        list_of_files['msd_id'] = list_of_files['msd_id'].apply(
            lambda entry: os.path.splitext(os.path.basename(entry))[0])

        if for_tracks is not None:
            list_of_files = list_of_files.merge(for_tracks, on=['msd_id'])

        for _, row in list_of_files.iterrows():
            if row['file_name'].endswith('.mp3'):
                temp_file = os.path.join(TMP_PATH, row['file_name'])

                try:
                    zip_object.extract(row['file_name'], TMP_PATH)
                    audio_time_series, sampling_rate = librosa.load(temp_file)
                    librosa_melspectrogram = librosa.feature.melspectrogram(
                        audio_time_series,
                        sampling_rate,
                        n_mels=n_mels,
                    )

                    if min_time is not None:
                        if librosa_melspectrogram.shape[1] < min_time:
                            logger.info('Sample to short.')
                            continue

                    if window is not None:
                        librosa_melspectrogram = window(
                            librosa_melspectrogram,
                            window_size,
                        )

                    result.append((
                        archive_file_name,
                        row['msd_id'],
                        librosa_melspectrogram,
                    ))
                except Exception as ex:
                    logger.exception(ex)

                try:
                    os.remove(temp_file)
                except OSError as ex:
                    logger.exception(ex)

                if len(result) % 1000 == 0:
                    logger.info('Looked at %s tracks.', len(result))

    return pd.DataFrame(result, columns=DATAFRAME_COLUMNS)


def merge_song_samples_with_features(data):
    """Merges extracted melspectrogram features.

    The song samples passed as an argument are merged with their respecting
    melspectrogram features.

    Args:
        data: the song samples passed as a pandas dataframe.

    Returns:
        A pandas dataframe containing the subset of samples, where the
        melspectrogram features are available.

    """
    dataset_columns = list(data.columns)
    dataset_columns.append('librosa_melspectrogram')
    logger.info(dataset_columns)
    dataset = pd.DataFrame(columns=dataset_columns)

    output_file_name_regex = OUTPUT_PREFIX + '_*.parquet'
    search_path = os.path.join(INTERIM_PATH, output_file_name_regex)
    feature_files = glob.glob(search_path)
    for feature_file in feature_files:
        features = pd.read_parquet(feature_file)
        samples = data.merge(features, on=['msd_id'])[dataset_columns]
        dataset = dataset.append(
            samples,
            sort=False,
            ignore_index=True,
        )

    return dataset


def use_center_window(data, window_size):
    """Extracts the center window of a given numpy array.

    Args:
        data: a two dimensional numpy array where the last dimension (axis=-1)
            should be windowed.
        window_size: window size of the last dimension.

    """
    size = data.shape[-1]
    start_idx = int((size - window_size) / 2)
    end_idx = start_idx + window_size

    return data[..., start_idx:end_idx]


@click.group()
@click.option('--debug', is_flag=True)
def cli(debug):
    """Sets up the click command group.

    Args:
        debug: a cli flag allwoing to enable debugging messages.

    """
    if debug:
        logging.loglevel(level=logging.DEBUG)
    else:
        logzero.loglevel(level=logging.INFO)

    warnings.simplefilter('ignore')


@cli.command()
@click.argument('chunk', type=int)
@click.argument('chunk_count', type=int)
def extract(chunk, chunk_count):
    """The cli extract command to extract melspectrogram features.

    Calling this command will extract all melspectrogram features for all mp3
    files contained in the provided archive files.

    It allows to split the list of archive files in multiple chunks to be able
    to process those chunks in parallel.

    Args:
        chunk: allows to select a chunk of archive files by giving a cli
            argument. This argument has to be the index of the chunk after
            spliting the list of archive files in chunk_count chunks.
        chunk_count: allows to specify in how many chunks the list of archive
            files gets split.

    """
    if chunk >= chunk_count:
        logger.error('Chunk needs to be smaller than chunk_count.')
        sys.exit(1)

    archive_files = glob.glob(os.path.join(MP3_ARCHIVE_PATH, '*.zip'))
    file_count = 0

    chunk_size = int(len(archive_files) / chunk_count)
    start_idx = chunk * chunk_size
    if chunk + 1 == chunk_count:
        end_idx = len(archive_files)
    else:
        end_idx = (chunk + 1) * chunk_size

    for zipfile_name in archive_files[start_idx:end_idx]:
        logger.info('Extracting: %s', zipfile_name)
        try:
            features = extract_mel_from_mp3s_in_zipfile(zipfile_name)
            file_count += features.shape[0]
            output_file_name = os.path.join(INTERIM_PATH, OUTPUT_PREFIX)
            archive_file_name = os.path.basename(zipfile_name)
            archive_file_name, _ = os.path.splitext(archive_file_name)
            output_file_name += ('_' + archive_file_name + '.parquet')
            features.to_parquet(output_file_name)

        except Exception as ex:
            logger.exception(ex)

    logger.info('Extracted featrues for %s tracks.', file_count)


def _extract_features(zipfile_name, dataset):
    """A wrapper function to be used as to parallelize the feature extraction.

    Args:
        zipfile_name: the name of the zipfile where data should be extracted.
        dataset: the dataframe to extract features for.

    """
    logger.info('Extracting: %s', zipfile_name)
    try:
        return extract_mel_from_mp3s_in_zipfile(
            zipfile_name,
            for_tracks=dataset['msd_id'],
            min_time=1200,  # routhly a 30 sec timeframe
            window=use_center_window,
            window_size=1200,  # routhly a 30 sec timeframe
        )
    except Exception as ex:
        logger.exception(ex)
        return None


@cli.command()
@click.argument('dataset-file', type=str)
@click.option(
    '--processes_count',
    '-p',
    default=None,
    type=int,
    help='Restrict the number of parallel processes that are used.',
)
def combine_with_dataset(dataset_file, processes_count):
    """Extracts and combines melspectrogram features with a given dataset.

    Args:
        dataset_file: path to a dataset file in csv format that is pandas
            readable.
        processes_count: the number of processes used for the multiprocessing
            pool.

    """
    if not os.path.isfile(dataset_file):
        logger.error('Specified dataset file does not exist')
        sys.exit(1)

    dataset = pd.read_csv(dataset_file, index_col=0)
    archive_files = glob.glob(os.path.join(MP3_ARCHIVE_PATH, '*.zip'))

    output_file_name, _ = os.path.splitext(os.path.basename(dataset_file))
    output_file_name += ('_' + OUTPUT_PREFIX + '.parquet')
    output_file_name = os.path.join(PROCESSED_PATH, output_file_name)

    extractor = functools.partial(_extract_features, dataset=dataset)

    with multiprocessing.Pool(processes=processes_count) as p:
        features = pd.concat(p.map(extractor, archive_files))
        dataset = dataset.merge(features, on=['msd_id'])
        dataset.to_parquet(output_file_name)

        logger.info('Extracted features for %d samples.', dataset.shape[0])


if __name__ == '__main__':
    cli()
