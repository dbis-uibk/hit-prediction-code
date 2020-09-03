"""Merges the msd mbids with lastfm info."""
import json

from logzero import logger
import pandas as pd
import pylast

data_path = 'data/hit_song_prediction_ismir2020'
dataset_name = '/interim/msd_mbid_lastfm.pickle'
lastfm_file = data_path + dataset_name


def get_api_config():
    """Loads the api config from the config file."""
    with open('lastfm_api_info.json') as file:
        return json.load(file)


def download_lastfm_info(mbids, lastfm_file, only_missing=True):
    """Downloads Last.fm info for songs in mbids.

    This function downloads Last.fm information for mbids contianed in the
    msd_mbid_map parameter.

    Args:
        mbids: mbids to pull information from Last.fm.
        lastfm_file: the file where the information should be stored.
        only_missing: If False, mbids already contained in the lastfm_file
            are overwritten. Otherwise, only the missing mbids are pulled
            form Last.fm.
    """
    api_info = get_api_config()
    network = pylast.LastFMNetwork(
        api_key=api_info['api_key'],
        api_secret=api_info['api_secret'],
    )

    columns = [
        'mbid',
        'lastfm_mbid',
        'lastfm_artist',
        'lastfm_artist_mbid',
        'lastfm_title',
        'lastfm_playcount',
        'lastfm_listener_count',
        'lastfm_duration',
        'lastfm_top_tags',
    ]

    mbids = set(mbids)

    if not only_missing:
        lastfm_info = pd.DataFrame(columns=columns)
    else:
        try:
            known_info = pd.read_pickle(lastfm_file)
            known_mbids = set(known_info['mbid'])

            lastfm_info = known_info
            mbids -= known_mbids
        except FileNotFoundError:
            pass

    for idx, mbid in enumerate(mbids):
        try:
            track = network.get_track_by_mbid(mbid)

            track_info = {
                'mbid':
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
        except pylast.MalformedResponseError as e:
            logger.exception(e)

    logger.info('Store Last.fm info for msd_mbid')
    lastfm_info.to_pickle(data_path + dataset_name)


def download_msd_mbid_lastfm():
    """Downloads Last.fm info for mibds in msd_mbid."""
    logger.info('Get msd_mbid map')
    msd_mbid_map = pd.read_csv(
        data_path + '/raw/msd-mbid-2016-01-results-ab.csv',
        names=['msd_id', 'mbid', 'title', 'artist'],
    )

    logger.info('Start Last.fm download')
    download_lastfm_info(msd_mbid_map['mbid'], lastfm_file, only_missing=True)


if __name__ == '__main__':
    download_msd_mbid_lastfm()
