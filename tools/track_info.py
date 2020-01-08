import json

import click

import requests

import pandas as pd

import musicbrainzngs

from dataset_creation import join, read_hits

RESULT_PATH = '.'

WIKIDATA_PATH = '/storage/nas3/datasets/wikipedia/wikidata'


@click.group()
@click.option('--path',
              default='.',
              help='The path where the results are stored.')
def cli(path):
    global RESULT_PATH
    RESULT_PATH = path


@cli.command()
@click.option('--dups',
              default=False,
              is_flag=True,
              help='Store duplicates in the resulting file.')
def musicbrainz(dups):
    musicbrainzngs.set_useragent('musicbrainz crawler', '1.0')
    WIKIDATA_PATH = '/storage/nas3/datasets/wikipedia/wikidata/'

    data = []
    num_of_match = 0
    num_of_miss = 0
    num_of_multi = 0

    charts = read_hits()
    artists = pd.read_csv(WIKIDATA_PATH + '/artist_dbpedia_wikidata.csv')
    artists = join(artists,
                   pd.read_csv(RESULT_PATH + '/artist_musicbrainz_id.csv'),
                   on=['artist_wikidata_uri'])
    charts = join(charts, artists,
                  on=['artist'])[['title', 'artist', 'musicbrainz_artist_id']]

    for i, song in charts.iterrows():
        title = song['title']
        arid = song['musicbrainz_artist_id']

        recordings = []

        results = musicbrainzngs.search_recordings(release=title, arid=arid)

        for recording in results['recording-list']:
            if recording['ext:score'] == '100' and recording['artist-credit']:
                for artist in recording['artist-credit']:
                    try:
                        if arid == artist['artist']['id']:
                            recordings.append(recording)
                    except Exception as ex:
                        print(ex)

        found = False
        if dups:
            if len(recordings) > 1:
                found = True
        else:
            if len(recordings) == 1:
                found = True
            elif len(recordings) >= 1:
                num_of_multi += 1

        if found:
            num_of_match += 1
            for recording in recordings:
                data.append({
                    'title': title,
                    'artist': song['artist'],
                    'musicbrainz_artist_id': arid,
                    'musicbrainz_recording_id': recording['id'],
                })
        else:
            num_of_miss += 1

        print(i + 1, '/', len(charts), 'found:', found, len(recordings), arid,
              title)

    print('Match', num_of_match, 'Multi', num_of_multi, 'Miss', num_of_miss)
    pd.DataFrame(data).to_csv(RESULT_PATH + '/msd_bb_matches_recording_id.csv')


@cli.command()
@click.option('--recording-file',
              default='msd_bb_matches_recording_id.csv',
              help='File containing the recordings.')
def acousticbrainz(recording_file):
    # recordings = pd.read_csv(recording_file)
    # TODO: split into chunks of 25
    recordings = [
        {
            'musicbrainz_recording_id': '8ac7f923-8418-4766-b3d7-406adb8aada1'
        },
        {
            'musicbrainz_recording_id': 'c7e5ae79-b20b-46ab-89b4-63612ac3206f'
        },
    ]

    recordings = pd.DataFrame(recordings)

    recording_id = recordings['musicbrainz_recording_id']
    hl = acousticbrainz_get_highlevel(recording_id)
    ll = acousticbrainz_get_lowlevel(recording_id)

    for key, value in hl.items():
        output_file = RESULT_PATH + '/' + key + '.highlevel.json'
        write_json_file(output_file, value)

    for key, value in ll.items():
        output_file = RESULT_PATH + '/' + key + '.lowlevel.json'
        write_json_file(output_file, value)


def acousticbrainz_get_highlevel(recording_ids):
    return acousticbrainz_get('high-level', recording_ids)


def acousticbrainz_get_lowlevel(recording_ids):
    return acousticbrainz_get('low-level', recording_ids)


def acousticbrainz_get(feature_type, recording_ids):
    BASE_URL = 'https://acousticbrainz.org/api/v1'

    url = BASE_URL + '/' + feature_type

    payload = {'recording_ids': ';'.join(recording_ids)}
    headers = {'content-type': 'application/json'}

    result = requests.get(url, params=payload, headers=headers)

    return result.json()


def write_json_file(file_name, data):
    with open(file_name, 'w') as outfile:
        json.dump(data, outfile)


if __name__ == '__main__':
    cli()
