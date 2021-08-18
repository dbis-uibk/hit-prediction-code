"""Wide and deep plan using all features."""
import os.path

from dbispipeline.evaluators import CvEpochEvaluator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import hit_prediction_code.common as common
from hit_prediction_code.dataloaders import ClassLoaderWrapper
from hit_prediction_code.dataloaders import EssentiaLoader
import hit_prediction_code.evaluations as evaluations
from hit_prediction_code.models.wide_and_deep import WideAndDeep
from hit_prediction_code.result_handlers import print_results_as_json

PATH_PREFIX = 'data/hit_song_prediction_ismir2020/processed'

dataloader = ClassLoaderWrapper(
    wrapped_loader=EssentiaLoader(
        dataset_path=os.path.join(
            PATH_PREFIX,
            'msd_bb_mbid_cleaned_matches_ab_unique.parquet',
        ),
        features=[
            *common.all_no_year_list(),
        ],
        label='weeks',
        nan_value=common.weeks_non_hit_value(),
    ),
    labels=common.weeks_binary_labels(),
)

pipeline = Pipeline([
    ('scale', MinMaxScaler()),
    ('model',
     WideAndDeep(
         epochs=common.wide_and_deep_epochs(),
         features=dataloader.wrapped_loader.feature_indices,
         batch_normalization=False,
         deep_activation='elu',
         dense_activation='elu',
         dropout_rate=0.1,
         loss='categorical_crossentropy',
         metrics=['mae', 'mse', 'categorical_crossentropy'],
         output_activation='softmax',
         label_output=True,
     )),
])

evaluator = CvEpochEvaluator(
    cv=evaluations.cv(),
    scoring=evaluations.metrics.classifier_scoring(),
    scoring_step_size=evaluations.scoring_step_size(),
)

result_handlers = [
    print_results_as_json,
]
