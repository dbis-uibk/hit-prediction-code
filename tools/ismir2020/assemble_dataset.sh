#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

python ${SCRIPT_DIR}/merge_msd_and_bb.py
python ${SCRIPT_DIR}/merge_msd_bb_and_mbid.py
python ${SCRIPT_DIR}/merge_msd_bb_mbid_and_ab.py
python ${SCRIPT_DIR}/merge_msd_bb_mbid_and_essentia.py
python ${SCRIPT_DIR}/merge_msd_bb_mbid_and_targets.py
python ${SCRIPT_DIR}/select_unique_features.py
python ${SCRIPT_DIR}/merge_msd_bb_mbid_dataset.py
