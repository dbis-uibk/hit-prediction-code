import dbispipeline.result_handlers as result_handlers
from dbispipeline.evaluators import GridEvaluator

from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

from dataloaders import MsdBbLoader


cv = KFold(n_splits=5, shuffle=True, random_state=42)

dataloader = MsdBbLoader(
        hits_file_path='/storage/nas3/datasets/music/billboard/msd_bb_matches.csv',  # noqa E501
        non_hits_file_path='/storage/nas3/datasets/music/billboard/msd_bb_non_matches.csv',  # noqa E501
        features_path='/storage/nas3/datasets/music/billboard',
        non_hits_per_hit=1,
        features=['hl'],
        label='weeks',
    )

pipeline = Pipeline([
    ('svm', SVR(gamma='scale')),
])

evaluator = GridEvaluator(
    parameters={
        'svm__C': [1.0],
    },
    grid_parameters={
        'verbose': 3,
        'cv': cv,
        'refit': False,
        'scoring': 'neg_mean_squared_error',
    },
)

result_handlers = [
    result_handlers.print_gridsearch_results,
]
