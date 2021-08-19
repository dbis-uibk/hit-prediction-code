"""FCN plan using mel-spect features."""
import os.path

from dbispipeline.evaluators import CvEpochEvaluator
from sklearn.pipeline import Pipeline

import hit_prediction_code.common as common
from hit_prediction_code.dataloaders import ClassLoaderWrapper
from hit_prediction_code.dataloaders import MelSpectLoader
import hit_prediction_code.evaluations as evaluations
from hit_prediction_code.models.cnn import FCN
from hit_prediction_code.result_handlers import print_results_as_json

PATH_PREFIX = 'data/hit_song_prediction_ismir2020/processed'

dataloader = ClassLoaderWrapper(
    wrapped_loader=MelSpectLoader(
        dataset_path=os.path.join(
            PATH_PREFIX,
            'msd_bb_mbid_cleaned_matches_melspect_db_unique.pickle',
        ),
        features='librosa_melspectrogram',
        label='peakPos',
        nan_value=common.peak_pos_non_hit_value(),
    ),
    labels=common.peak_position_binary_labels(),
)

pipeline = Pipeline([
    ('model',
     FCN(
         epochs=common.fcn_epochs(),
         loss='categorical_crossentropy',
         metrics=['mae', 'mse', 'categorical_crossentropy'],
         output_activation='softmax',
         label_output=True,
     )),
])

evaluator = CvEpochEvaluator(
    cv=evaluations.cv(),
    scoring=evaluations.metrics.classifier_scoring(),
    scoring_step_size=evaluations.fcn_scoring_step_size(),
)

result_handlers = [
    print_results_as_json,
]
