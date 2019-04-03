import dbispipeline.result_handlers as result_handlers
from dbispipeline.evaluators import GridEvaluator

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from dataloaders import MsdBbLoader

cv = KFold(n_splits=5, shuffle=True, random_state=42)

dataloader = MsdBbLoader(
    hits_file_path='/storage/nas3/datasets/music/billboard/msd_bb_matches.csv',
    non_hits_file_path=
    '/storage/nas3/datasets/music/billboard/msd_bb_non_matches.csv',
    features_path='/storage/nas3/datasets/music/billboard',
    non_hits_per_hit=1,
    features=['hl', 'year'],
    label='peak',
    nan_value=101,
    random_state=42,
)

pipeline = Pipeline([
    ('scale', MinMaxScaler()),
    ('logreg', LogisticRegression(multi_class='auto', solver='lbfgs')),
])

evaluator = GridEvaluator(
    parameters={
        'logreg__C': [1.0],
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
