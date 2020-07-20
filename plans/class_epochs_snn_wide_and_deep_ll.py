"""SNN Wide and deep model epoch evaluation plan using lowlevel features."""
from dbispipeline.evaluators import CvEpochEvaluator
import dbispipeline.result_handlers
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import hit_prediction_code.common as common
from hit_prediction_code.dataloaders import EssentiaLoader
import hit_prediction_code.evaluations as evaluations
from hit_prediction_code.models.wide_and_deep import WideAndDeep

dataloader = EssentiaLoader(
    dataset_path='data/processed/msd_bb_balanced_essentia.pickle',
    features=[
        *common.ll_list(),
    ],
    label='peak',
    nan_value=150,
    binarize_labels=True,
)

pipeline = Pipeline([
    ('scale', MinMaxScaler()),
    ('model',
     WideAndDeep(
         epochs=1000,
         features=dataloader.feature_indices,
         batch_normalization=False,
         deep_activation='selu',
         dense_activation='selu',
         output_activation='softmax',
         num_dense_layer=7,
         dropout_rate=0.1,
         loss='binary_crossentropy',
         metrics=['binary_crossentropy'],
         label_output=True,
     )),
])

evaluator = CvEpochEvaluator(
    cv=evaluations.cv(),
    scoring=evaluations.metrics.classifier_scoring(),
    scoring_step_size=10,
)

result_handlers = [
    dbispipeline.result_handlers.print_gridsearch_results,
]
