import dbispipeline.result_handlers as result_handlers
from dbispipeline.evaluators import GridEvaluator

from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

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
        *common.ll_filterd_list(),
    ],
    label='peak',
    nan_value=150,
    random_state=42,
)

pipeline = Pipeline([
    ('scale', MinMaxScaler()),
    ('linreg', ElasticNet(normalize=False)),
])

evaluator = GridEvaluator(
    parameters={},
    grid_parameters=evaluations.grid_parameters(),
)

result_handlers = [
    result_handlers.print_gridsearch_results,
]
