"""Linear regression model evaluation plan using year features only."""
from dbispipeline.evaluators import GridEvaluator
import dbispipeline.result_handlers as result_handlers
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from hit_prediction_code.dataloaders import EssentiaLoader
import hit_prediction_code.evaluations as evaluations

dataloader = EssentiaLoader(
    dataset_path=
    'data/hit_song_prediction_ismir2020/processed/msd_bb_balanced_essentia.pickle',
    features=[
        ('year', 'wide'),
    ],
    label='peak',
    nan_value=150,
)

pipeline = Pipeline([
    ('scale', MinMaxScaler()),
    ('linreg', LinearRegression()),
])

evaluator = GridEvaluator(
    parameters={},
    grid_parameters=evaluations.grid_parameters(),
)

result_handlers = [
    result_handlers.print_gridsearch_results,
]
