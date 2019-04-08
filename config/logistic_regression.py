import dbispipeline.result_handlers as result_handlers
from dbispipeline.evaluators import GridEvaluator

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from dataloaders import MsdBbLoader

from evaluations import metrics


cv = KFold(n_splits=5, shuffle=True, random_state=42)

dataloader = MsdBbLoader(
    hits_file_path='/storage/nas3/datasets/music/billboard/msd_bb_matches.csv',
    non_hits_file_path=
    '/storage/nas3/datasets/music/billboard/msd_bb_non_matches.csv',
    features_path='/storage/nas3/datasets/music/billboard',
    non_hits_per_hit=1,
    features=[('genre', 'wide'), ('mood', 'wide'), ('voice', 'wide'),
              ('year', 'wide')],
    label='peak',
    nan_value=150,
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
        'scoring': metrics.scoring(),
        'return_train_score': True,
    },
)

result_handlers = [
    result_handlers.print_gridsearch_results,
]
