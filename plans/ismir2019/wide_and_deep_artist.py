import dbispipeline.result_handlers as result_handlers
from dbispipeline.evaluators import GridEvaluator

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import hit_prediction_code.common as common

from hit_prediction_code.dataloaders import MsdBbLoader

import hit_prediction_code.evaluations as evaluations

from hit_prediction_code.models.wide_and_deep import WideAndDeep

dataloader = MsdBbLoader(
    hits_file_path='data/processed/msd_bb_matches.csv',
    non_hits_file_path='data/processed/msd_bb_non_matches.csv',
    features_path='data/processed',
    non_hits_per_hit=1,
    features=[
        ('artist', 'wide'),
        ('year', 'wide'),
    ],
    label='peak',
    nan_value=150,
    random_state=42,
)

pipeline = Pipeline([
    ('scale', MinMaxScaler()),
    ('model', WideAndDeep(features=dataloader.feature_indices)),
])

evaluator = GridEvaluator(
    parameters={
        'model__batch_normalization': [False],
        'model__dropout_rate': [0.25],
        'model__epochs': [250, 300, 400, 500],
    },
    grid_parameters=evaluations.grid_parameters(),
)

result_handlers = [
    result_handlers.print_gridsearch_results,
]
