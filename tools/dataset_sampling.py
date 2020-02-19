#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os.path
import sys

import click
import pandas as pd

LOGGER = logging.getLogger(__name__)
PROCESSED_PATH = os.path.join('data', 'processed')


@click.group()
@click.option('--debug', is_flag=True)
def cli(debug):
    """Sets up the click command group.

    Args:
        debug: a cli flag allwoing to enable debugging messages.

    """
    if debug:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.disable()


@cli.command()
@click.argument('hit-samples', type=str)
@click.argument('non-hit-samples', type=str)
@click.argument('output-file', type=str)
@click.option(
    '--non-hits-per-hit',
    '-n',
    default=None,
    type=float,
    help='Number of non-hit samples per hit sample in the final dataset.',
)
@click.option(
    '--random-state',
    default=None,
    type=int,
    help='Specify a random state if you want reproducability.',
)
def combine(hit_samples, non_hit_samples, output_file, non_hits_per_hit,
            random_state):
    """The cli combine command merges a hit and non-hit sub set into a dataset.

    Args:
        hit_samples: path to the file containing the hit samples which are all
            used.
        non_hit_samples: path to the file containing the non-hit samples.
        output_file: path to the output file.
        non_hits_per_hit: how many non-hit samples are used per hit sample. If
            this remains unspecified, then all non-hit samples are used. If the
            resulting number of non-hits would exceed the total number of
            non-hits all non-hits are used.
        random_state: the random state used for sampling.

    """
    hits = pd.read_pickle(hit_samples)
    LOGGER.info("%d hit samples loaded.", len(hits))

    non_hits = pd.read_pickle(non_hit_samples)
    LOGGER.info("%d non-hit samples loaded.", len(non_hits))

    if non_hits_per_hit is not None:
        num_of_samples = int(len(hits) * non_hits_per_hit)

        if num_of_samples < len(non_hits):
            non_hits = non_hits.sample(
                n=num_of_samples,
                random_state=random_state,
            )
            LOGGER.info('Sampled %d non-hit samples', len(non_hits))

    output = pd.concat([hits, non_hits], sort=False, ignore_index=True)
    output.to_pickle(output_file)
    LOGGER.info('Stored a dataset containing %d samples.', len(output))


@cli.command()
@click.argument('samples-file', type=str)
@click.argument('output-file', type=str)
def add_essentia_features(samples_file, output_file):
    """The add-essentia-features command adds essentia features for samples.

    It is required that the files msd_bb_ll_features.h5 and
    msd_bb_hl_features.h5 are present in the processed data directory which is
    by default located at data/processed.

    Args:
        samples_file: path to the file containing the samples.
        output_file: path to the output file.

    """
    data = pd.read_pickle(samples_file)
    LOGGER.info("%d samples loaded.", len(data))

    hl_features_file = os.path.join(PROCESSED_PATH, 'msd_bb_hl_features.h5')
    hl_features = pd.read_hdf(hl_features_file)
    data = data.merge(hl_features, on=['msd_id'])
    LOGGER.info("%d samples combined with highlevel features.", len(data))

    ll_features_file = os.path.join(PROCESSED_PATH, 'msd_bb_ll_features.h5')
    ll_features = pd.read_hdf(ll_features_file)
    data = data.merge(ll_features, on=['msd_id'])
    LOGGER.info("%d samples combined with lowlevel features.", len(data))

    data.to_pickle(output_file)
    LOGGER.info('Stored a dataset containing %d samples.', len(data))


if __name__ == '__main__':
    cli()
