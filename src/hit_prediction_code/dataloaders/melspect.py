"""Module to handle mel spectrograms."""
import functools
import glob
import multiprocessing
import os
import tempfile
import zipfile

import librosa
from logzero import logger
import pandas as pd

INTERIM_PATH = os.path.join(
    'data',
    'hit_song_prediction_ismir2020',
    'interim',
)
MP3_ARCHIVE_PATH = os.path.join(
    'data',
    'millionsongdataset',
    'mp3s',
)
OUTPUT_PREFIX = 'melspect_features'
TMP_PATH = tempfile.gettempdir()

DATAFRAME_COLUMNS = [
    'archive_file_name',
    'msd_id',
    'librosa_melspectrogram',
]


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


def merge_song_samples_with_features(data, project_home='.'):
    """Merges extracted melspectrogram features.

    The song samples passed as an argument are merged with their respecting
    melspectrogram features.

    Args:
        data: the song samples passed as a pandas dataframe.
        project_home: path to the data folder.

    Returns:
        A pandas dataframe containing the subset of samples, where the
        melspectrogram features are available.

    """
    dataset_columns = list(data.columns)
    dataset_columns.append('librosa_melspectrogram')
    logger.info(dataset_columns)
    dataset = pd.DataFrame(columns=dataset_columns)

    output_file_name_regex = OUTPUT_PREFIX + '_*.pickle'
    search_path = os.path.join(
        project_home,
        INTERIM_PATH,
        output_file_name_regex,
    )
    feature_files = glob.glob(search_path)
    for feature_file in feature_files:
        features = pd.read_pickle(feature_file)
        samples = data.merge(features, on=['msd_id'])[dataset_columns]
        dataset = dataset.append(
            samples,
            sort=False,
            ignore_index=True,
        )

    return dataset


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


def combine_with_dataset(dataset,
                         dataset_filename,
                         processes_count=None,
                         project_home='.',
                         output_file_prefix=None,
                         compression='xz'):
    """Extracts and combines melspectrogram features with a given dataset.

    Args:
        dataset: dataset to extract melspects for msd_id column.
        dataset_filename: name of the dataset file used to base the output
            filename on.
        processes_count: the number of processes used for the multiprocessing
            pool.
        project_home: path to the data folder.
        output_file_prefix: If not None, the given prefix is used instead of
            the basename of the dataset_filename.
        compression: used to store the pickle.
    """
    archive_files = glob.glob(
        os.path.join(
            project_home,
            MP3_ARCHIVE_PATH,
            '*.zip',
        ))

    if output_file_prefix is not None:
        output_filename = output_file_prefix
    else:
        output_filename, _ = os.path.splitext(
            os.path.basename(dataset_filename))
    file_extention = '.pickle'
    if compression is not None:
        file_extention += ('.' + compression)
    output_filename += ('_' + OUTPUT_PREFIX)
    output_filename = os.path.join(
        project_home,
        os.path.dirname(dataset_filename),
        output_filename,
    )

    extractor = functools.partial(_extract_features, dataset=dataset)

    with multiprocessing.Pool(processes=processes_count) as p:
        features = pd.concat(p.map(extractor, archive_files))
        dataset = dataset.merge(features, on=['msd_id'])

        chunk_size = 50_000

        if dataset.shape[0] > chunk_size:
            columns = [
                c for c in dataset.columns if 'librosa_melspectrogram' not in c
            ]
            dataset[columns].to_pickle(output_filename + file_extention,
                                       compression)
            logger.info('Stored metadata for features of %d samples.',
                        dataset.shape[0])

            for count, chunk_start in enumerate(
                    range(0, dataset.shape[0], chunk_size)):
                chunk_end = chunk_start + chunk_size
                chunk_name = '_' + str(count)

                data = dataset[chunk_start:chunk_end]
                data.to_pickle(output_filename + chunk_name + file_extention,
                               compression)
                logger.info('Stored feature chunk %d containing %d samples.',
                            count, data.shape[0])
        else:
            dataset.to_pickle(output_filename + file_extention, compression)
        logger.info('Extracted features for %d samples.', dataset.shape[0])


def extract(chunk, chunk_count, project_home='.'):
    """Function to extract melspectrogram features.

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
        project_home: path to the data folder.

    """
    assert chunk < chunk_count

    archive_files = glob.glob(
        os.path.join(
            project_home,
            MP3_ARCHIVE_PATH,
            '*.zip',
        ))
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
            output_file_name = os.path.join(
                project_home,
                INTERIM_PATH,
                OUTPUT_PREFIX,
            )
            archive_file_name = os.path.basename(zipfile_name)
            archive_file_name, _ = os.path.splitext(archive_file_name)
            output_file_name += ('_' + archive_file_name + '.pickle')
            features.to_pickle(output_file_name)

        except Exception as ex:
            logger.exception(ex)

    logger.info('Extracted featrues for %s tracks.', file_count)
