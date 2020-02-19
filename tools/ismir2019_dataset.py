#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Cli tool to download our ISMIR 2019 Hit Song Prediction dataset."""

import hashlib
import json
import logging
import multiprocessing as mp
import os
from shutil import copyfile
import sys
import tarfile
import urllib.request

import click
import numpy as np
import pandas as pd
import requests

RAW_PATH = os.path.join('data', 'raw')
INTERIM_PATH = os.path.join('data', 'interim')
PROCESSED_PATH = os.path.join('data', 'processed')

LOGGER = logging.getLogger(__name__)


@click.group()
def cli():
    """Sets up the click command group."""
    pass


@cli.command()
def download_raw_data():
    """Downloads the dataset from Zenodeo.

    The `Hit Song Prediction <https://doi.org/10.5281/zenodo.3258042>`_ is
    available on Zenodo.

    """
    get_zenodo_record_meta(3258042)


@cli.command()
def unpack_features():
    """Unpacks the features needed to prepare data for the models.

    Ensure that you first run the download_raw_data command.

    """
    feature_file_name = os.path.join(RAW_PATH, 'msd_audio_features.tar.gz')
    with tarfile.open(feature_file_name) as feature_file:
        feature_file.extractall(path=INTERIM_PATH)


@cli.command()
def prepare_data():
    """Prepares the data needed to run the models.

    Ensure that you first run the unpack_features command.

    """
    copyfile(
        src=os.path.join(RAW_PATH, 'msd_bb_matches.csv'),
        dst=os.path.join(PROCESSED_PATH, 'msd_bb_matches.csv'),
    )
    copyfile(
        src=os.path.join(RAW_PATH, 'msd_bb_non_matches.csv'),
        dst=os.path.join(PROCESSED_PATH, 'msd_bb_non_matches.csv'),
    )

    features = _combine_features(_combine_ll_features)
    lowlevel_features_file = os.path.join(
        PROCESSED_PATH,
        'msd_bb_ll_features.h5',
    )
    features.to_hdf(lowlevel_features_file, 'll')

    features = _combine_features(_combine_hl_features)
    highlevel_features_file = os.path.join(
        PROCESSED_PATH,
        'msd_bb_hl_features.h5',
    )
    features.to_hdf(highlevel_features_file, 'hl')


@cli.command()
@click.pass_context
def get(ctx):
    """Runs the download_raw_data, unpack_features and prepare_data command."""
    ctx.invoke(download_raw_data)
    ctx.invoke(unpack_features)
    ctx.invoke(prepare_data)


def _combine_features(combine_function):
    hits_file = os.path.join(RAW_PATH, 'msd_bb_matches.csv')
    hits = set(pd.read_csv(hits_file)['msd_id'])

    non_hits_file = os.path.join(RAW_PATH, 'msd_bb_non_matches.csv')
    non_hits = set(pd.read_csv(non_hits_file)['msd_id'])

    msd_ids = hits | non_hits

    all_features = pd.DataFrame()

    df_split = np.array_split(list(msd_ids), mp.cpu_count() * 4)
    with mp.Pool() as pool:
        features = pool.imap_unordered(combine_function, df_split)

        for feature in features:
            all_features = all_features.append(feature,
                                               sort=False,
                                               ignore_index=True)

        return all_features

    return None


def _load_features(msd_ids, feature_file_suffix, features_path=INTERIM_PATH):
    features = pd.DataFrame()
    for msd_id in msd_ids:
        try:
            file_id = pd.DataFrame([msd_id], columns=['msd_id'])
            feature = pd.io.json.json_normalize(
                _load_feature(features_path, msd_id, feature_file_suffix))
            features = features.append(file_id.join(feature),
                                       sort=False,
                                       ignore_index=True)
        except FileNotFoundError as error:
            LOGGER.exception(error)

    return features


def _combine_ll_features(msd_ids):
    return _load_features(msd_ids, '.lowlevel.json')


def _combine_hl_features(msd_ids):
    return _load_features(msd_ids, '.mp3.highlevel.json')


def _load_feature(features_path, msd_id, file_suffix):
    file_prefix = '/features_tracks_' + msd_id[2].lower() + '/'
    file_name = features_path + file_prefix + msd_id + file_suffix

    with open(file_name) as features:
        return json.load(features)


def get_zenodo_record_meta(record_id, path=RAW_PATH):
    """
    Downloads meta file from zenodo for a given record id.

    Args:
        record_id: the record_id to fetch.
        path: the destination path where the files should be stored.

    Returns:
        A JSON formated meta information file.

    """
    api_base_url = 'https://zenodo.org/api/records/'

    try:
        res = requests.get(api_base_url + str(record_id), timeout=15.)
    except requests.exceptions.ConnectTimeout:
        LOGGER.info('Connection timeout.')
        sys.exit(1)
    except Exception as ex:
        LOGGER.exception(ex)
        LOGGER.info('Connection error.')
        sys.exit(2)

    if res.ok:
        payload = json.loads(res.text)
        files = payload['files']
        total_size = sum(f['size'] for f in files)

        checksum_file = os.path.join(RAW_PATH, 'md5sums.txt')
        with open(checksum_file, 'wt') as md5file:
            for f in files:
                fname = f['key']
                checksum = f['checksum'].split(':')[-1]
                md5file.write(f'{checksum}  {fname}\n')

        LOGGER.info('Title: %s', payload['metadata']['title'])
        LOGGER.info('Keywords: %s',
                    (', '.join(payload['metadata'].get('keywords', []))))
        LOGGER.info('Publication date: %s',
                    str(payload['metadata']['publication_date']))
        LOGGER.info('DOI: %s', payload['metadata']['doi'])
        LOGGER.info('Total size: %.1f MB', total_size / 2**20)

        for f in files:
            link = f['links']['self']
            size = f['size'] / 2**20
            LOGGER.info('Link: %s   size: %.1f MB', str(link), size)
            filename = os.path.join(path, f['key'])
            checksum = f['checksum']

            if is_file_valid(filename, checksum):
                LOGGER.info('%s is already downloaded correctly.', fname)
                continue

            urllib.request.urlretrieve(url=link, filename=filename)

            if is_file_valid(filename, checksum):
                LOGGER.info('Checksum is correct.')
            else:
                LOGGER.info('Checksum is INCORRECT!')
                LOGGER.info('  File is deleted.')
                os.remove(filename)

        LOGGER.info('All files have been downloaded.')
    else:
        LOGGER.info('Record could not get accessed.')
        sys.exit(1)


def is_file_valid(filename, checksum):
    """Checks if a file is valid by comparing actual and given checksum.

    Args:
        filename: the filename to check.
        checksum: the expected checksum.

    Returns:
        True if the computed and given checksum are the same and otherwise
        False.

    """
    blocksize = 65536

    algorithm, hash_value = checksum.split(':')
    if not os.path.exists(filename):
        return False

    file_hash = hashlib.new(algorithm)
    with open(filename, 'rb') as f:
        chunk = f.read(blocksize)
        while chunk:
            file_hash.update(chunk)
            chunk = f.read(blocksize)

    return hash_value == file_hash.hexdigest()


if __name__ == '__main__':
    cli()
