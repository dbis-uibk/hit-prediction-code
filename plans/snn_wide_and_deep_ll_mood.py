"""SNN Wide and deep model evaluation plan using mood and lowlevel features."""
from dbispipeline.evaluators import GridEvaluator
import dbispipeline.result_handlers as result_handlers
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import hit_prediction_code.common as common
from hit_prediction_code.dataloaders import EssentiaLoader
import hit_prediction_code.evaluations as evaluations
from hit_prediction_code.models.wide_and_deep import WideAndDeep

dataloader = EssentiaLoader(
    dataset_path='data/hit_song_prediction_ismir2020/processed/msd_bb_balanced_essentia.pickle',
    features=[
        *common.all_ll_mood_list(),
    ],
    label='peak',
    nan_value=150,
)

pipeline = Pipeline([
    ('scale', MinMaxScaler()),
    ('model', WideAndDeep(features=dataloader.feature_indices)),
])

evaluator = GridEvaluator(
    parameters={
        'model__epochs': [1000],
        'model__batch_normalization': [False],
        'model__deep_activation': ['selu'],
        'model__dense_activation': ['selu'],
        'model__output_activation': ['elu'],
        'model__num_dense_layer': [7],
        'model__dropout_rate': [0.1],
    },
    grid_parameters=evaluations.grid_parameters(),
)

result_handlers = [
    result_handlers.print_gridsearch_results,
]
