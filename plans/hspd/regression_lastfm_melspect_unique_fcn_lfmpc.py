"""FCN plan using mel-spect features."""
import os.path

from dbispipeline.evaluators import CvEpochEvaluator
from sklearn.pipeline import Pipeline

import hit_prediction_code.common as common
from hit_prediction_code.dataloaders import MelSpectLoader
import hit_prediction_code.evaluations as evaluations
from hit_prediction_code.models.cnn import FCN
from hit_prediction_code.result_handlers import print_results_as_json

PATH_PREFIX = 'data/hit_song_prediction_lastfm/processed'

dataloader = MelSpectLoader(
    dataset_path=os.path.join(
        PATH_PREFIX,
        'msd_lastfm_matches_melspect_db_features_unique.pickle',
    ),
    features='librosa_melspectrogram',
    label='lastfm_playcount',
    nan_value=0,
)

pipeline = Pipeline([
    ('model', FCN(epochs=common.fcn_epochs())),
])

evaluator = CvEpochEvaluator(
    cv=evaluations.cv(),
    scoring=evaluations.metrics.scoring(
        hit_nonhit_accuracy_score=None,
        categories=common.lfmpc_labels(),
    ),
    scoring_step_size=evaluations.fcn_scoring_step_size(),
)

result_handlers = [
    print_results_as_json,
]
