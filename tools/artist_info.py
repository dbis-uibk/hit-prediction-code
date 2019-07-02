import click
import platform
import re

import matplotlib.pyplot as plt

import pandas as pd

import requests
from qwikidata.sparql import return_sparql_query_results

from dataset_creation import read_msd_unique_artists, read_hits, read_non_hits, join


RESULT_PATH = '.'


@click.group()
@click.option(
    '--path', default='.', help='The path where the results are stored.')
def cli(path):
    global RESULT_PATH
    RESULT_PATH = path


@cli.command()
def artists():
    hits = read_hits()
    artist_id = read_msd_unique_artists()
    artist_id = artist_id.dropna(subset=['mb_artist_id'])
    # artists = hits['artist'].map(artist_preprocess).reset_index(name='artist')
    # hits = pd.concat([hits.drop('artist', axis=1), artists], axis=1)
    num_of_hits = len(hits)

    hits = join(hits, artist_id, on=['artist'])
    print(hits['mb_artist_id'])
    print(num_of_hits, len(hits))

    hits = hits.groupby(['artist']).size().sort_values(ascending=False).reset_index(name='counts')

    hits = hits[hits['artist'].str.contains('[0-9]', regex=True)]

    # print(hits)
    hits.plot()
    plt.show()


def artist_preprocess(name):
    name = str(name)
    name = name.lower()
    # name = re.sub(r'[^a-zA-Z ]', '', name)

    return name


@cli.command()
def wikidata():
    query_string = """
    SELECT $WDid
    WHERE {
        ?WDid (wdt:P279)* wd:Q4022
    }
    """
    query_string = """
    SELECT $WDid
    WHERE {
      wd:Q4022 (wdt:P279)* ?WDid
    }
    """
    url = 'http://query.wikidata.org/bigdata/namespace/wdq/sparql'
    results = requests.get(url, params={'query': query_string, 'format': 'json',})

    print(results.text)


if __name__ == '__main__':
    cli()
