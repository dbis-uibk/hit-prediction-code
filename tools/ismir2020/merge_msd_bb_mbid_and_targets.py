"""Merges dataset with target labels."""
import os

from logzero import logger
import pandas as pd

from hit_prediction_code.dataloaders import matcher

path_prefix = 'data/hit_song_prediction_ismir2020/interim/'


def target_filter(data, target_col, keep_lowest):
    """Filters duplicates."""
    return matcher.filter_duplicates(
        data=data,
        id_cols=['uuid'],
        target_col=target_col,
        keep_lowest=keep_lowest,
    )


def read_msd_bb_mbid(dataset):
    """Gets msd bb mbid map for a dataset."""
    return pd.read_csv(
        os.path.join(
            path_prefix,
            'msd_bb_mbid_' + dataset + '.csv',
        ),
        header=0,
        index_col=0,
    )


logger.info('Read msd bb mbid info')
data = read_msd_bb_mbid('cleaned_matches')
data = data.append(read_msd_bb_mbid('cleaned_matches'))
data = data.append(read_msd_bb_mbid('cleaned_matches'))
data.drop_duplicates(inplace=True)

logger.info('Read lastfm info')
lastfm = pd.read_pickle(path_prefix + 'msd_mbid_lastfm.pickle')[[
    'mbid',
    'lastfm_mbid',
    'lastfm_listener_count',
    'lastfm_playcount',
]]
lastfm = data.merge(lastfm, on=['mbid'])
lc = target_filter(
    lastfm[['uuid', 'lastfm_listener_count']],
    target_col='lastfm_listener_count',
    keep_lowest=False,
).drop_duplicates()
pc = target_filter(
    lastfm[['uuid', 'lastfm_playcount']],
    target_col='lastfm_playcount',
    keep_lowest=False,
).drop_duplicates()
lastfm_data = lc.merge(pc, on=['uuid'])

logger.info('Read chart data')
charts = pd.read_csv(
    path_prefix + 'msd_bb_cleaned_matches.csv',
    header=0,
    index_col=0,
)[[
    'msd_id',
    'peakPos',
    'weeks',
    'rank',
]]
charts = charts.append(
    pd.read_csv(
        path_prefix + 'msd_bb_exact_matches.csv',
        header=0,
        index_col=0,
    )[[
        'msd_id',
        'peakPos',
        'weeks',
    ]])
charts.drop_duplicates(inplace=True)

charts = data.merge(charts, on=['msd_id'])
peak = target_filter(
    charts[['uuid', 'peakPos']],
    target_col='peakPos',
    keep_lowest=True,
).drop_duplicates()
weeks = target_filter(
    charts[['uuid', 'weeks']],
    target_col='weeks',
    keep_lowest=False,
).drop_duplicates()
chart_data = peak.merge(weeks, on=['uuid'])

data = lastfm_data.merge(chart_data, on=['uuid'], how='left')

logger.info('Store targets')
data.to_csv(path_prefix + 'msd_bb_mbid_targets.csv')
