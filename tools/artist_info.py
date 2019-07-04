import click

import matplotlib.pyplot as plt

import pandas as pd

from SPARQLWrapper import SPARQLExceptions, SPARQLWrapper, JSON

from qwikidata.entity import WikidataItem, WikidataProperty
from qwikidata.linked_data_interface import get_entity_dict_from_api
from qwikidata.json_dump import WikidataJsonDump

from dataset_creation import join, read_hits, read_msd_unique_artists, read_non_hits

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
    hit_artists = pd.read_csv('/storage/nas3/datasets/wikipedia/wikidata/artist_dbpedia_wikidata_hits.csv') 
    artist_id = read_msd_unique_artists()
    artist_id = artist_id.dropna(subset=['mb_artist_id'])
    num_of_hits = len(hits)

    hits_mbid = join(hits, artist_id, on=['artist'])
    hits_wd = join(hits, hit_artists, on=['artist'])
    hit_artists = join(hit_artists, artist_id, on=['artist'])
    hits = join(hits, hit_artists, on=['artist'])

    print(num_of_hits, len(hits_mbid), len(hits_wd), len(hits))

    hits = hits.groupby([
        'artist',
    ]).size().sort_values(ascending=False).reset_index(name='counts')

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

    artists = read_non_hits().drop_duplicates(subset=['artist'])

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
    data = {}
    non_hit_artists = pd.read_csv('/storage/nas3/datasets/wikipedia/wikidata/artist_dbpedia_wikidata_non_hits.csv')

    for artist in non_hit_artists['artist_wikidata_uri']:
        artist_data = get_entity_dict_from_api(artist.rsplit('/', 1)[-1])
        data[artist] = artist_data

    pd.DataFrame(data).to_json(RESULT_PATH + '/artist_non_hits_data.json')


if __name__ == '__main__':
    cli()
