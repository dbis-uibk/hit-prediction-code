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

WIKIDATA_PATH = '/storage/nas3/datasets/wikipedia/wikidata'


@click.group()
@click.option(
    '--path', default='.', help='The path where the results are stored.')
def cli(path):
    global RESULT_PATH
    RESULT_PATH = path


@cli.command()
@click.option('--hits', default=False, is_flag=True, help='Only hits.')
@click.option('--non-hits', default=False, is_flag=True, help='Only non-hits.')
def artists(hits, non_hits):
    if hits:
        songs = read_hits()
    elif non_hits:
        songs = read_non_hits()
    else:
        songs = read_songs()

    all_artists = set(songs['artist'])
    mapped_artists = set(read_artist_dbpedia_wikidata()['artist'])
    no_info = all_artists - mapped_artists

    print(no_info, len(all_artists), len(mapped_artists), len(no_info))


def artist_preprocess(name):
    name = str(name)
    name = name.lower()
    # name = re.sub(r'[^a-zA-Z ]', '', name)

    return name


def read_songs():
    songs = read_hits()
    songs = songs.append(read_non_hits(), sort=False)
    songs.drop_duplicates(subset=['artist'], inplace=True)

    return songs


def read_artist_dbpedia_wikidata():
    return pd.read_csv(WIKIDATA_PATH + '/artist_dbpedia_wikidata.csv')


def missing_artists():
    all_artists = set(read_songs()['artist'])
    mapped_artists = set(read_artist_dbpedia_wikidata()['artist'])

    return all_artists - mapped_artists


@cli.command()
@click.option(
    '--get-all',
    default=False,
    is_flag=True,
    help='Loads all artists by default only missing artists are loaded.')
def dbpedia(get_all):
    dest_file = RESULT_PATH + '/artist_dbpedia_wikidata.csv'
    mapping = []
    failed = []

    if get_all:
        artists = read_songs()['artist']
        df = pd.DataFrame()
    else:
        artists = missing_artists()
        df = pd.read_csv(dest_file, index_col=0)

    num_of_artists = len(artists)
    for i, artist in enumerate(artists, 1):
        skip = False

        for string in ['ft.', 'feat.', 'featuring']:
            if string in artist.lower():
                skip = True

        if not skip:
            entry = {}
            entry['artist'] = artist
            try:
                result, _ = get_artist_from_dbpedia(entry['artist'])
            except SPARQLExceptions.QueryBadFormed as ex:
                print(ex)
                failed.append({'artist': artist, 'exception': str(ex)})
                continue

            if result:
                entry['artist_dbpedia_uri'] = result['item']['value']
                entry['artist_wikidata_uri'] = result['same']['value']
                mapping.append(entry)

            print('Track:', i, '/', num_of_artists, '    found:', bool(result),
                  '    ', artist)

        if i % 100 == 0 and len(mapping):
            df = df.append(mapping, ignore_index=True)
            df.to_csv(dest_file)
            mapping = []

    df.append(mapping, ignore_index=True).to_csv(dest_file)
    pd.DataFrame(failed).to_csv('failed_requests.csv')


def get_artist_from_dbpedia(artist):
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setTimeout(300)
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
            UNION
            { ?item rdf:type yago:Singer110599806 .}
            UNION
            { ?item rdf:type yago:Musician110340312 .}
            UNION
            { ?item rdf:type yago:Musician110339966 .}
            UNION
            { ?item rdf:type yago:Artist109812338 .}
            UNION
            { ?item rdf:type yago:MusicalOrganization108246613 . }
            UNION
            { ?item rdf:type yago:Group100031264 . }
            ?item rdfs:label ?label .
            FILTER (lang(?label) = 'en')
            FILTER (regex(str(?label), "%s", "i"))
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
    artists = pd.read_csv(WIKIDATA_PATH + '/artist_dbpedia_wikidata.csv')

    for artist in artists['artist_wikidata_uri']:
        artist_data = get_entity_dict_from_api(artist.rsplit('/', 1)[-1])
        data[artist] = artist_data

    pd.DataFrame(data).to_json(RESULT_PATH + '/artist_non_hits_data.json')


if __name__ == '__main__':
    cli()
