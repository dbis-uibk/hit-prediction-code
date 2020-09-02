"""Merges the msd mbids with lastfm info."""
import json

from logzero import logger
import pandas as pd
import pylast

data_path = 'data/hit_song_prediction_ismir2020'
dataset_name = '/interim/msd_mbid_lastfm.pickle'

with open('lastfm_api_info.json') as file:
    api_info = json.load(file)

network = pylast.LastFMNetwork(
    api_key=api_info['api_key'],
    api_secret=api_info['api_secret'],
)

logger.info('Merge msd_bb_matches with mbid')
msd_mbid_map = pd.read_csv(
    data_path + '/raw/msd-mbid-2016-01-results-ab.csv',
    names=['msd_id', 'mbid', 'title', 'artist'],
)

mbids = set(msd_mbid_map['mbid'])

columns = [
    'msd_mbid',
    'lastfm_mbid',
    'lastfm_artist',
    'lastfm_artist_mbid',
    'lastfm_title',
    'lastfm_playcount',
    'lastfm_listener_count',
    'lastfm_duration',
    'lastfm_top_tags',
]
lastfm_info = pd.DataFrame(columns=columns)
for idx, mbid in enumerate(mbids):
    try:
        track = network.get_track_by_mbid(mbid)

        track_info = {
            'msd_mbid':
                mbid,
            'lastfm_mbid':
                track.get_mbid(),
            'lastfm_artist':
                track.get_artist().get_name(),
            'lastfm_artist_mbid':
                track.get_artist().get_mbid(),
            'lastfm_title':
                track.get_title(),
            'lastfm_playcount':
                track.get_playcount(),
            'lastfm_listener_count':
                track.get_listener_count(),
            'lastfm_duration':
                track.get_duration(),
            'lastfm_top_tags': [{
                'tag_name': tag[0].get_name(),
                'tag_count': tag[1],
            } for tag in track.get_top_tags()],
        }

        lastfm_info = lastfm_info.append(
            track_info,
            ignore_index=True,
        )

        if (idx + 1) % 50 == 0:
            logger.info('Store intermediate result with %d items' %
                        lastfm_info.shape[0])
            lastfm_info.to_pickle(data_path + dataset_name)
    except pylast.WSError as e:
        if e.get_id() == '6':
            logger.warning(e)
        else:
            logger.error('Status: %s' % e.get_id())
            logger.exception(e)
            break

logger.info('Store Last.fm info for msd_mbid')
lastfm_info.to_pickle(data_path + dataset_name)
