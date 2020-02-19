"""Linear regression model evaluation plan using mood and lowlevel features."""
from dbispipeline.evaluators import GridEvaluator
import dbispipeline.result_handlers as result_handlers
import hit_prediction_code.common as common
from hit_prediction_code.dataloaders import EssentiaLoader
import hit_prediction_code.evaluations as evaluations
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

dataloader = EssentiaLoader(
    dataset_path='data/processed/msd_bb_balanced_essentia.pickle',
    features=[
        *common.all_ll_mood_list(),
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
