import dbispipeline.result_handlers as result_handlers
from dbispipeline.evaluators import GridEvaluator

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor

import common

from dataloaders import MsdBbLoader

import evaluations

dataloader = MsdBbLoader(
    hits_file_path='/storage/nas3/datasets/music/billboard/msd_bb_matches.csv',
    non_hits_file_path=
    '/storage/nas3/datasets/music/billboard/msd_bb_non_matches.csv',
    features_path='/storage/nas3/datasets/music/billboard',
    non_hits_per_hit=1,
    features=[
        ('artist', 'deep'),
        ('year', 'wide'),
    ],
    label='peak',
    nan_value=150,
    random_state=42,
)

pipeline = Pipeline([
    ('scale', MinMaxScaler()),
    ('model', GradientBoostingRegressor),
])

evaluator = GridEvaluator(
    parameters={
        'model__n_estimators': [100, 500],
    },
    grid_parameters=evaluations.grid_parameters(),
)

result_handlers = [
    result_handlers.print_gridsearch_results,
]
