import dbispipeline.result_handlers as result_handlers
from dbispipeline.evaluators import GridEvaluator
import hit_prediction_code.common as common
from hit_prediction_code.dataloaders import EssentiaLoader
import hit_prediction_code.evaluations as evaluations
from hit_prediction_code.models.wide_and_deep import WideAndDeep
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

dataloader = EssentiaLoader(
    dataset_path='data/processed/msd_bb_balanced_essentia.pickle',
    features=[
        *common.mood_list(),
    ],
    label='peak',
    nan_value=150,
)

pipeline = Pipeline([
    ('scale', MinMaxScaler()),
    ('wide_and_deep', WideAndDeep(features=dataloader.feature_indices)),
])

evaluator = GridEvaluator(
    parameters={
        'wide_and_deep__epochs': [10, 50, 100, 200],
        'wide_and_deep__batch_normalization': [False, True],
        'wide_and_deep__dropout_rate': [None, 0.25, 0.5],
    },
    grid_parameters=evaluations.grid_parameters(),
)

result_handlers = [
    result_handlers.print_gridsearch_results,
]
