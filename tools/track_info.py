import click

import musicbrainzngs


RESULT_PATH = '.'

WIKIDATA_PATH = '/storage/nas3/datasets/wikipedia/wikidata'


@click.group()
@click.option(
    '--path', default='.', help='The path where the results are stored.')
def cli(path):
    global RESULT_PATH
    RESULT_PATH = path


@cli.command()
def musicbrainz():
    fields = {
        'arid': '0b30341b-b59d-4979-8130-b66c0e475321',
        'release': 'Old Town Road',
    }
    musicbrainzngs.set_useragent('musicbrainz crawler', '1.0')
    print(musicbrainzngs.search_recordings(**fields))


if __name__ == '__main__':
    cli()
