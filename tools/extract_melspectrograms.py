#!/usr/bin/env python
"""Command line tool to extract melspectrogram features for a given dataset."""
import logging
import os
import sys
import warnings

import click
import logzero
from logzero import logger

from hit_prediction_code.dataloaders import melspect


@click.group()
@click.option('--debug', is_flag=True)
def cli(debug):
    """Sets up the click command group.

    Args:
        debug: a cli flag allwoing to enable debugging messages.

    """
    if debug:
        logzero.loglevel(level=logging.DEBUG)
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

    melspect.extract(chunk, chunk_count)


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

    melspect.combine_with_dataset(dataset_file, processes_count)


if __name__ == '__main__':
    cli()
