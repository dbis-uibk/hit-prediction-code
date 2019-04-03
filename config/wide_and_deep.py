import dbispipeline.result_handlers as result_handlers
from dbispipeline.evaluators import GridEvaluator

from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from dataloaders import MsdBbLoader

from models.wide_and_deep import WideAndDeep

cv = KFold(n_splits=5, shuffle=True, random_state=42)

dataloader = MsdBbLoader(
    hits_file_path='/storage/nas3/datasets/music/billboard/msd_bb_matches.csv',
    non_hits_file_path=
    '/storage/nas3/datasets/music/billboard/msd_bb_non_matches.csv',
    features_path='/storage/nas3/datasets/music/billboard',
    non_hits_per_hit=1,
    features=[('hl', 'wide'), ('year', 'wide')],
    label='peak',
    nan_value=101,
    random_state=42,
)

pipeline = Pipeline([
    ('scale', MinMaxScaler()),
    ('wide_and_deep', WideAndDeep(features=dataloader.feature_indices)),
])

evaluator = GridEvaluator(
    parameters={
        'wide_and_deep__epochs': [10, 100, 500],
    },
    grid_parameters={
        'verbose':
        3,
        'cv':
        cv,
        'refit':
        False,
        'scoring': [
            'explained_variance',
            'neg_mean_absolute_error',
            'neg_mean_squared_error',
            'neg_median_absolute_error',
            'r2',
        ],
    },
)

result_handlers = [
    result_handlers.print_gridsearch_results,
]
