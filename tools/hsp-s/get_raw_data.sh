#!/bin/bash

curl ftp://ftp.acousticbrainz.org/pub/acousticbrainz/acousticbrainz-labs/download/msdtombid/msd-mbid-2016-01-results-ab.csv.bz2 | bzip2 -d > data/hit_song_prediction_ismir2020/raw/msd-mbid-2016-01-results-ab.csv

curl -o data/hit_song_prediction_ismir2020/raw/msd_bb_non_matches.csv https://zenodo.org/record/3258042/files/msd_bb_non_matches.csv
