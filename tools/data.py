"""Tool to manage data."""
import os

import click
from logzero import logger
import yaml

LINK_CONFIG_FILE = 'data/links.yaml'


@click.group()
def cli():
    """Cli for the data tool."""


@cli.command()
@click.option(
    '--dataset-path-prefix',
    '-p',
    default='/storage/nas3/datasets',
    type=str,
    help='The path prefix used to find the datasets.',
)
def link(dataset_path_prefix):
    """Links the needed datasets to the dataset folder.

    The configuration is loaded from `data/links.yaml`.

    Args:
        dataset_path_prefix: the prefix used for the dataset paths.
    """
    logger.info('Linking datasets from: %s' % dataset_path_prefix)

    try:
        with open(LINK_CONFIG_FILE, 'r') as yml:
            try:
                config = yaml.safe_load(yml)
            except yaml.YAMLError as ex:
                logger.exception(ex)
    except FileNotFoundError:
        logger.error('No such file or directory: %s' % LINK_CONFIG_FILE)
        exit(1)

    try:
        for dataset in config['datasets']:
            _link_dataset(dataset_path_prefix, dataset)
    except KeyError:
        logger.error('Config does not contain datasets.')
        exit(3)


def _link_dataset(dataset_path_prefix, dataset):
    logger.info('\t- %s' % dataset)

    src = os.path.join(dataset_path_prefix, dataset)
    if not os.path.isdir(src):
        logger.warn('Source is not directory: %s' % src)
        return

    dst = 'data'
    if not os.path.isdir(dst):
        logger.error('No destination data directory: %s' % dst)
        exit(2)

    dataset = os.path.join(*dataset.split('/')[1:])
    dst = os.path.join(dst, dataset)

    try:
        os.symlink(src, dst, target_is_directory=True)
    except FileExistsError:
        if not os.readlink(dst) == src:
            logger.warn('Wrong source: \'%s\' instead of \'%s\'' %
                        (os.readlink(dst), src))


if __name__ == '__main__':
    cli()
