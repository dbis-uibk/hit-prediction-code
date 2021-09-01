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

PATH_PREFIX = 'data/hit_song_prediction_msd_bb_lfm_ab/processed'

dataloader = ClassLoaderWrapper(
    wrapped_loader=MelSpectLoader(
        dataset_path=os.path.join(
            PATH_PREFIX,
            'hsp-s_melspect.pickle',
        ),
        features='librosa_melspectrogram_db',
        label='weeks',
        nan_value=common.weeks_non_hit_value(),
    ),
    labels=common.weeks_binary_labels(),
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
