"""Simple NN model plan using ll and hl features for listener count."""
from dbispipeline.evaluators import CvEpochEvaluator
import dbispipeline.result_handlers
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import hit_prediction_code.common as common
from hit_prediction_code.dataloaders import EssentiaLoader
import hit_prediction_code.evaluations as evaluations
from hit_prediction_code.models.simple_nn import SimpleNN

dataloader = EssentiaLoader(
    dataset_path='data/processed/lfm_popularity_dataset_20.pickle',
    features=[
        *common.mood_list(),
        *common.genre_list(),
        *common.voice_list(),
        *common.ll_list(),
    ],
    label='listener_count',
    nan_value=0,
)

pipeline = Pipeline([
    ('scale', MinMaxScaler()),
    ('model',
     SimpleNN(
         epochs=500,
         deep_activation='selu',
         dense_activation='selu',
         output_activation='elu',
         dense_sizes=(256, 128, 128, 128, 128, 64),
         dropout_rate=0.1,
     )),
])

evaluator = CvEpochEvaluator(
    cv=evaluations.cv(),
    scoring=evaluations.metrics.scoring(),
    scoring_step_size=10,
)

result_handlers = [
    dbispipeline.result_handlers.print_gridsearch_results,
]
