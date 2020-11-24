#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

python ${SCRIPT_DIR}/merge_msd_and_lastfm.py
python ${SCRIPT_DIR}/merge_msd_lastfm_and_ab.py
python ${SCRIPT_DIR}/merge_msd_lastfm_and_essentia.py
