import json

import click

import pandas as pd

import musicbrainzngs

from dataset_creation import join, read_hits

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
    musicbrainzngs.set_useragent('musicbrainz crawler', '1.0')
    WIKIDATA_PATH = '/storage/nas3/datasets/wikipedia/wikidata/'

    data = []
    num_of_match = 0
    num_of_miss = 0
    num_of_multi = 0

    charts = read_hits()
    artists = pd.read_csv(WIKIDATA_PATH + '/artist_dbpedia_wikidata.csv')
    artists = join(
        artists,
        pd.read_csv(RESULT_PATH + '/artist_musicbrainz_id.csv'),
        on=['artist_wikidata_uri'])
    charts = join(
        charts, artists,
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
        if len(recordings) == 1:
            num_of_match += 1
            found = True
            data.append({
                'title': title,
                'artist': song['artist'],
                'musicbrainz_artist_id': arid,
                'musicbrainz_recording_id': recordings[0]['id'],
            })
        elif len(recordings) < 1:
            num_of_miss += 1
        else:
            num_of_multi += 1

        print(i + 1, '/', len(charts), 'found:', found, len(recordings), arid,
              title)

    print('Match', num_of_match, 'Multi', num_of_multi, 'Miss', num_of_miss)
    pd.DataFrame(data).to_csv(RESULT_PATH + '/msd_bb_matches_recording_id.csv')


if __name__ == '__main__':
    cli()
