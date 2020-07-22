"""Merges the ismir2019 dataset with mbids."""
import pandas as pd

msd_mbid_map = pd.read_csv(
    'data/hit_song_prediction_ismir2020/raw/msd-mbid-2016-01-results-ab.csv',
    names=['msd_id', 'mbid', 'title', 'artist'],
)

# merge mateches
matches = pd.read_csv(
    'data/hit_song_prediction_ismir2019/msd_bb_matches.csv',
    index_col=0,
)

matches_mbid = matches.merge(
    msd_mbid_map,
    on=['msd_id'],
    suffixes=('_msd_bb', '_msd_mbid'),
)
matches_mbid.to_csv(
    'data/hit_song_prediction_ismir2020/interim/msd_bb_mbid_matches.csv')

# merge non-mateches
non_matches = pd.read_csv(
    'data/hit_song_prediction_ismir2019/msd_bb_non_matches.csv',
    index_col=0,
)

non_matches_mbid = non_matches.merge(
    msd_mbid_map,
    on=['msd_id'],
    suffixes=('_msd_bb', '_msd_mbid'),
)
non_matches_mbid.to_csv(
    'data/hit_song_prediction_ismir2020/interim/msd_bb_mbid_non_matches.csv')
