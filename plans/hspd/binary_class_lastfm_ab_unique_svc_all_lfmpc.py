"""SVC plan using all features."""
import os.path

from dbispipeline.evaluators import CvEpochEvaluator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import hit_prediction_code.common as common
from hit_prediction_code.dataloaders import BinaryClassLoaderWrapper
from hit_prediction_code.dataloaders import EssentiaLoader
import hit_prediction_code.evaluations as evaluations
from hit_prediction_code.models.svm import SVC
from hit_prediction_code.result_handlers import print_results_as_json

PATH_PREFIX = 'data/hit_song_prediction_msd_bb_lfm_ab/processed'

dataloader = BinaryClassLoaderWrapper(wrapped_loader=EssentiaLoader(
    dataset_path=os.path.join(
        PATH_PREFIX,
        'hsp-l_acousticbrainz.parquet',
    ),
    features=[
        *common.all_no_year_list(),
    ],
    label='lastfm_playcount',
    nan_value=0,
))

pipeline = Pipeline([
    ('scale', MinMaxScaler()),
    ('model', SVC()),
])

evaluator = CvEpochEvaluator(
    cv=evaluations.cv(),
    scoring=evaluations.metrics.classifier_scoring(),
    scoring_step_size=1,
)

result_handlers = [
    print_results_as_json,
]
