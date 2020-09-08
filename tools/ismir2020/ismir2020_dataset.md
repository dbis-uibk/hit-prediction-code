# Create the ISMIR 2020 Dataset

It is required to have the MSD and the billboard charts.

In a first step all matches are determined by running: `merge_msd_and_bb.py`
In addition the non-matches from ISMIR 2019 are used. Further, we need the
msd-mbid map and lastfm info. The lastfm info can be downloaded by executing:
`merge_msd_mbid_and_lastfm.py`

After that we merge the hits and non-hits with mbid and lastfm to remove
duplicates. This is done by running: `merge_msd_bb_and_mbid.py`

After that, we need to extract the features and targest. This is done by
running the following scripts.
* `merge_msd_bb_mbid_and_ab.py`
* `merge_msd_bb_mbid_and_essentia.py`
* `merge_msd_bb_mbid_and_targets.py`

To get a singe feature set per song we need to select the "best fitting ones"
by running:
`select_unique_features.py`

*Start Missing*
After that we need to extract the mel-spects for all known essentia features.
* `merge_msd_bb_mbid_essentia_and_melspect.py`
*End Missing*

Final, we combine the whole dataset by running: `merge_msd_bb_mbid_dataset.py`
