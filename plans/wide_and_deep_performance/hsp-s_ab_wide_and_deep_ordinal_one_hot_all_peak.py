"""Wide and deep model evaluation plan using all features."""
import os.path

from dbispipeline.evaluators import CvEpochEvaluator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import hit_prediction_code.common as common
from hit_prediction_code.dataloaders import EssentiaLoader
import hit_prediction_code.evaluations as evaluations
from hit_prediction_code.models.wide_and_deep import WideAndDeepOrdinal
from hit_prediction_code.result_handlers import print_results_as_json

PATH_PREFIX = 'data/hit_song_prediction_msd_bb_lfm_ab/processed'

dataloader = EssentiaLoader(
    dataset_path=os.path.join(
        PATH_PREFIX,
        'hsp-s_acousticbrainz.parquet',
    ),
    features=[
        *common.all_no_year_list(),
    ],
    label='peakPos',
    nan_value=common.peak_pos_non_hit_value(),
)

pipeline = Pipeline([
    ('scale', MinMaxScaler()),
    ('model',
     WideAndDeepOrdinal(
         epochs=common.wide_and_deep_epochs(),
         features=dataloader.feature_indices,
         batch_normalization=False,
         deep_activation='elu',
         dense_activation='elu',
         dropout_rate=0.1,
         predict_strategy='argmax',
         vectorization_strategy='one_hot',
         loss='categorical_crossentropy',
         metrics=['mse', 'categorical_crossentropy'],
         output_activation='softmax',
         labels=common.peak_position_labels(),
     )),
])

evaluator = CvEpochEvaluator(
    cv=evaluations.cv(),
    scoring=evaluations.metrics.scoring(
        categories=common.peak_position_labels()),
    scoring_step_size=evaluations.scoring_step_size(),
)

result_handlers = [
    print_results_as_json,
]
