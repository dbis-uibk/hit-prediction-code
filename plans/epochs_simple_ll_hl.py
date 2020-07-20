"""Simple NN model plan using ll and hl features for peak chart position."""
from dbispipeline.evaluators import CvEpochEvaluator
import dbispipeline.result_handlers
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import hit_prediction_code.common as common
from hit_prediction_code.dataloaders import EssentiaLoader
import hit_prediction_code.evaluations as evaluations
from hit_prediction_code.models.simple_nn import SimpleNN

dataloader = EssentiaLoader(
    dataset_path='data/hit_song_prediction_ismir2020/processed/msd_bb_balanced_essentia.pickle',
    features=[
        *common.mood_list(),
        *common.genre_list(),
        *common.voice_list(),
        *common.ll_list(),
    ],
    label='peak',
    nan_value=150,
)

pipeline = Pipeline([
    ('scale', MinMaxScaler()),
    ('model',
     SimpleNN(
         epochs=1000,
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
