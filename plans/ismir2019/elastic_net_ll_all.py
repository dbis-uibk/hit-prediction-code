"""Plan for ISMIR2019."""
from dbispipeline.evaluators import GridEvaluator
import dbispipeline.result_handlers as result_handlers
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import hit_prediction_code.common as common
from hit_prediction_code.dataloaders import MsdBbLoader
import hit_prediction_code.evaluations as evaluations

dataloader = MsdBbLoader(
    hits_file_path='data/hit_song_prediction_ismir2019/processed/msd_bb_matches.csv',
    non_hits_file_path='data/hit_song_prediction_ismir2019/processed/msd_bb_non_matches.csv',
    features_path='data/processed',
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
