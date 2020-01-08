#!/usr/bin/env python
# -*- coding: utf-8 -*-

import hashlib
import json
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


@click.group()
def cli():
    pass


@cli.command()
def raw_data():
    get_zenodo_record_meta(3258042)


@cli.command()
def interim_data():
    with tarfile.open('data/raw/msd_audio_features.tar.gz') as f:
        f.extractall(path='data/interim')


@cli.command()
def prepare_data():
    copyfile(
        src='data/raw/msd_bb_matches.csv',
        dst='data/processed/msd_bb_matches.csv',
    )
    copyfile(
        src='data/raw/msd_bb_non_matches.csv',
        dst='data/processed/msd_bb_non_matches.csv',
    )

    combine_highlevel_features()
    combine_lowlevel_features()


@cli.command()
@click.pass_context
def get(ctx):
    ctx.invoke(raw_data)
    ctx.invoke(interim_data)
    ctx.invoke(prepare_data)


def combine_lowlevel_features():
    features = _combine_features(_combine_ll_features)
    features.to_hdf('data/processed/msd_bb_ll_features.h5', 'll')


def combine_highlevel_features():
    features = _combine_features(_combine_hl_features)
    features.to_hdf('data/processed/msd_bb_hl_features.h5', 'hl')


def _combine_features(combine_function):
    hits = set(pd.read_csv('data/raw/msd_bb_matches.csv')['msd_id'])
    non_hits = set(pd.read_csv('data/raw/msd_bb_non_matches.csv')['msd_id'])
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


def _combine_ll_features(msd_ids):
    features_path = 'data/interim'

    ll_features = pd.DataFrame()
    for msd_id in msd_ids:
        try:
            file_id = pd.DataFrame([msd_id], columns=['msd_id'])
            feature = pd.io.json.json_normalize(
                _get_lowlevel_feature(features_path, msd_id))
            ll_features = ll_features.append(file_id.join(feature),
                                             sort=False,
                                             ignore_index=True)
        except FileNotFoundError as error:
            print(error)

    return ll_features


def _combine_hl_features(msd_ids):
    features_path = 'data/interim'

    hl_features = pd.DataFrame()
    for msd_id in msd_ids:
        try:
            file_id = pd.DataFrame([msd_id], columns=['msd_id'])
            feature = pd.io.json.json_normalize(
                _get_highlevel_feature(features_path, msd_id))
            hl_features = hl_features.append(file_id.join(feature),
                                             sort=False,
                                             ignore_index=True)
        except FileNotFoundError as error:
            print(error)

    return hl_features


def _get_highlevel_feature(features_path, msd_id):
    file_suffix = '.mp3.highlevel.json'
    return _load_feature(features_path, msd_id, file_suffix)


def _get_lowlevel_feature(features_path, msd_id):
    file_suffix = '.lowlevel.json'
    return _load_feature(features_path, msd_id, file_suffix)


def _load_feature(features_path, msd_id, file_suffix):
    file_prefix = '/features_tracks_' + msd_id[2].lower() + '/'
    file_name = features_path + file_prefix + msd_id + file_suffix

    with open(file_name) as features:
        return json.load(features)


def get_zenodo_record_meta(record_id, path='data/raw'):
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
        eprint('Connection timeout.')
        sys.exit(1)
    except Exception as ex:
        eprint(ex)
        eprint('Connection error.')
        sys.exit(2)

    if res.ok:
        payload = json.loads(res.text)
        files = payload['files']
        total_size = sum(f['size'] for f in files)

        with open('data/raw/md5sums.txt', 'wt') as md5file:
            for f in files:
                fname = f['key']
                checksum = f['checksum'].split(':')[-1]
                md5file.write(f'{checksum}  {fname}\n')

        eprint('Title: {}'.format(payload['metadata']['title']))
        eprint('Keywords: ' +
               (', '.join(payload['metadata'].get('keywords', []))))
        eprint('Publication date: ' + payload['metadata']['publication_date'])
        eprint('DOI: ' + payload['metadata']['doi'])
        eprint('Total size: {:.1f} MB'.format(total_size / 2**20))

        for f in files:
            link = f['links']['self']
            size = f['size'] / 2**20
            eprint()
            eprint(f'Link: {link}   size: {size:.1f} MB')
            filename = os.path.join(path, f['key'])
            checksum = f['checksum']

            if check_file_hash(filename, checksum):
                eprint(f'{fname} is already downloaded correctly.')
                continue

            urllib.request.urlretrieve(url=link, filename=filename)

            if check_file_hash(filename, checksum):
                eprint(f'Checksum is correct.')
            else:
                eprint(f'Checksum is INCORRECT!')
                eprint('  File is deleted.')
                os.remove(filename)

        eprint('All files have been downloaded.')
    else:
        eprint('Record could not get accessed.')
        sys.exit(1)


def check_file_hash(filename, checksum):
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


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


if __name__ == '__main__':
    cli()
