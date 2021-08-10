"""LogisticRegressionClassifier plan using mean and std mel-spect features."""
import os.path

from dbispipeline.evaluators import CvEpochEvaluator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import hit_prediction_code.common as common
from hit_prediction_code.dataloaders import ClassLoaderWrapper
from hit_prediction_code.dataloaders import MelSpectMeanStdLoader
import hit_prediction_code.evaluations as evaluations
from hit_prediction_code.models.linear import LogisticRegressionClassifier
from hit_prediction_code.result_handlers import print_results_as_json

PATH_PREFIX = 'data/hit_song_prediction_ismir2020/processed'

dataloader = ClassLoaderWrapper(
    wrapped_loader=MelSpectMeanStdLoader(
        dataset_path=os.path.join(
            PATH_PREFIX,
            'msd_bb_mbid_cleaned_matches_melspect_unique.pickle.xz',
        ),
        features='librosa_melspectrogram',
        label='peakPos',
        nan_value=common.peak_pos_non_hit_value(),
    ),
    labels=common.peak_position_binary_labels(),
)

pipeline = Pipeline([
    ('scale', MinMaxScaler()),
    ('model', LogisticRegressionClassifier()),
])

evaluator = CvEpochEvaluator(
    cv=evaluations.cv(),
    scoring=evaluations.metrics.classifier_scoring(),
    scoring_step_size=1,
)

result_handlers = [
    print_results_as_json,
]
