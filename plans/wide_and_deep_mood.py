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
    ('model', WideAndDeep(features=dataloader.feature_indices)),
])

evaluator = GridEvaluator(
    parameters={
        'model__epochs': [10, 25, 50, 100, 200, 300],
        'model__batch_normalization': [False],
        'model__dense_activation': ['elu'],
        'model__output_activation': ['elu'],
        'model__dropout_rate': [0.1],
    },
    grid_parameters=evaluations.grid_parameters(),
)

result_handlers = [
    result_handlers.print_gridsearch_results,
]
