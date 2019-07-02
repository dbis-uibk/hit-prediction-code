import click
import platform
import re

import matplotlib.pyplot as plt

import pandas as pd

from SPARQLWrapper import SPARQLExceptions, SPARQLWrapper, JSON 

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
def dbpedia():
    mapping = []

    artists = read_hits().drop_duplicates(subset=['artist'])

    for artist in artists['artist']:
        entry = {}
        entry['artist'] = artist
        try:
            result = get_artist_from_dbpedia(entry['artist'])
        except SPARQLExceptions.QueryBadFormed:
            continue

        if result:
            print(result)

            entry['artist_dbpedia_uri'] = result['item']['value']
            entry['artist_wikidata_uri'] = result['same']['value']
            mapping.append(entry)

    pd.DataFrame(mapping).to_csv(RESULT_PATH + '/artist_dbpedia_wikidata.csv')


def get_artist_from_dbpedia(artist):
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setQuery("""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX purl: <http://purl.org/linguistics/gold/>
        SELECT DISTINCT ?item ?same
        WHERE
        {
            { ?item purl:hypernym dbr:Singer . }
            UNION
            { ?item purl:hypernym dbr:Band . }
            ?item rdfs:label ?label .
            FILTER (lang(?label) = 'en')
            FILTER (str(?label) = '""" + artist + """')
            ?item owl:sameAs ?same .
            FILTER (regex(?same, 'wikidata.org'))
        } LIMIT 2
    """)
    sparql.setReturnFormat(JSON)
    results = sparql.query()

    print(results.info())
    results = results.convert()["results"]["bindings"]
    print(len(results))

    if len(results) == 1:
        return results[0]
    else:
        return None


@cli.command()
def wikidata():
    pass
    # sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    # sparql.setQuery("""
    # SELECT ?item ?itemLabel
    # WHERE
    # {
        # ?item wdt:P31 wd:Q146 .
        # SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
    # }
    # """)
    # sparql.setReturnFormat(JSON)
    # results = sparql.query()
    # print(results.info())

    # results = results.convert()

    # results_df = pd.io.json.json_normalize(results['results']['bindings'])
    # results_df[['item.value', 'itemLabel.value']].head()


if __name__ == '__main__':
    cli()
