"""LinearRegression model plan using mean and std mel-spect features."""
import os.path

from dbispipeline.evaluators import CvEpochEvaluator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import hit_prediction_code.common as common
from hit_prediction_code.dataloaders import MelSpectMeanStdLoader
import hit_prediction_code.evaluations as evaluations
from hit_prediction_code.models.linear import LinearRegression
from hit_prediction_code.result_handlers import print_results_as_json

PATH_PREFIX = 'data/hit_song_prediction_lastfm/processed'

dataloader = MelSpectMeanStdLoader(
    dataset_path=os.path.join(
        PATH_PREFIX,
        'msd_lastfm_matches_melspect_features_unique.pickle',
    ),
    features='librosa_melspectrogram',
    label='peakPos',
    nan_value=common.peak_pos_non_hit_value(),
)

pipeline = Pipeline([
    ('scale', MinMaxScaler()),
    ('model', LinearRegression()),
])

evaluator = CvEpochEvaluator(
    cv=evaluations.cv(),
    scoring=evaluations.metrics.scoring(
        categories=common.peak_position_labels()),
    scoring_step_size=1,
)

result_handlers = [
    print_results_as_json,
]
