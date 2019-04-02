import dbispipeline.result_handlers as result_handlers
from dbispipeline.evaluators import GridEvaluator

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from dataloaders import MsdBbLoader


dataloader = MsdBbLoader(
        hits_file_path='/storage/nas3/datasets/music/billboard/msd_bb_matches.csv',  # noqa E501
        non_hits_file_path='/storage/nas3/datasets/music/billboard/msd_bb_non_matches.csv',  # noqa E501
        features_path='/storage/nas3/datasets/music/billboard',
        non_hits_per_hit=1,
        features=['hl'],
        label='weeks',
    )

pipeline = Pipeline([
    ('logres', LogisticRegression()),
])

evaluator = GridEvaluator(
    parameters={
        'logres__C': [1.0],
    },
    grid_parameters={
        'verbose': 3,
        'cv': 5,
        'refit': False,
        'scoring': 'neg_mean_squared_error',
    },
)

result_handlers = [
    result_handlers.print_gridsearch_results,
]
