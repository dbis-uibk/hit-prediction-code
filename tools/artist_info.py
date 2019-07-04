from urllib import parse

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
    hit_artists = pd.read_csv(
        '/storage/nas3/datasets/wikipedia/wikidata/artist_dbpedia_wikidata_hits.csv'
    )
    artist_id = read_msd_unique_artists()
    artist_id = artist_id.dropna(subset=['mb_artist_id'])
    # num_of_hits = len(hits)

    # hits_mbid = join(hits, artist_id, on=['artist'])
    # hits_wd = join(hits, hit_artists, on=['artist'])
    # hit_artists = join(hit_artists, artist_id, on=['artist'])
    # hits = join(hits, hit_artists, on=['artist'])

    # print(num_of_hits, len(hits_mbid), len(hits_wd), len(hits))

    hits = hits.groupby([
        'artist',
    ]).size().sort_values(ascending=False).reset_index(name='counts')

    no_info = set(hits['artist']) - set(hit_artists['artist'])

    print(len(no_info), no_info)
    # print(hits)
    # hits.plot()
    # plt.show()


def artist_preprocess(name):
    name = str(name)
    name = name.lower()
    # name = re.sub(r'[^a-zA-Z ]', '', name)

    return name


@cli.command()
def dbpedia():
    dest_file = RESULT_PATH + '/artist_dbpedia_wikidata.csv'
    mapping = []

    artists = read_hits()
    artists = artists.append(read_non_hits(), sort=False)
    artists.drop_duplicates(subset=['artist'], inplace=True)

    num_of_artists = len(artists)
    for i, artist in enumerate(artists['artist'], 1):
        entry = {}
        entry['artist'] = artist
        try:
            result, _ = get_artist_from_dbpedia(entry['artist'])
        except SPARQLExceptions.QueryBadFormed as ex:
            print(ex)
            continue

        if result:
            entry['artist_dbpedia_uri'] = result['item']['value']
            entry['artist_wikidata_uri'] = result['same']['value']
            mapping.append(entry)

        print('Track:', i, '/', num_of_artists, '    found:', bool(result),
              '    ', artist)

        if i % 100 == 0:
            pd.DataFrame(mapping).to_csv(dest_file)

    pd.DataFrame(mapping).to_csv(dest_file)


def get_artist_from_dbpedia(artist):
    # artist = parse.quote(artist, safe=' ')
    print(artist)

    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setQuery("""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT DISTINCT ?item ?same
        WHERE
        {
            { ?item rdf:type schema:MusicGroup . }
            UNION
            { ?item rdf:type dbo:MusicalArtist . }
            UNION
            { ?item rdf:type dbo:Band . }
            ?item rdfs:label ?label .
            FILTER (lang(?label) = 'en')
            FILTER (str(?label) = "%s")
            ?item owl:sameAs ?same .
            FILTER (regex(?same, 'wikidata.org'))
        } LIMIT 2
    """ % artist)
    sparql.setReturnFormat(JSON)
    results = sparql.query()
    info = results.info()

    results = results.convert()["results"]["bindings"]

    if len(results) == 1:
        return results[0], info
    else:
        return None, info


@cli.command()
def wikidata():
    data = {}
    non_hit_artists = pd.read_csv(
        '/storage/nas3/datasets/wikipedia/wikidata/artist_dbpedia_wikidata_non_hits.csv'
    )

    for artist in non_hit_artists['artist_wikidata_uri']:
        artist_data = get_entity_dict_from_api(artist.rsplit('/', 1)[-1])
        data[artist] = artist_data

    pd.DataFrame(data).to_json(RESULT_PATH + '/artist_non_hits_data.json')


if __name__ == '__main__':
    cli()
