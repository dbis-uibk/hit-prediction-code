"""CRNN model evaluation plan."""
from dbispipeline.evaluators import GridEvaluator
import dbispipeline.result_handlers as result_handlers
from hit_prediction_code.dataloaders import MelSpectLoader
import hit_prediction_code.evaluations as evaluations
from hit_prediction_code.models.crnn import CRNNModel
from sklearn.pipeline import Pipeline

dataloader = MelSpectLoader(
    dataset_path='data/processed/msd_bb_balanced_dev_sample.pickle',
    features='librosa_melspectrogram',
    label='peak',
    nan_value=150,
    random_state=42,
)

pipeline = Pipeline([
    ('model', CRNNModel()),
])

evaluator = GridEvaluator(
    parameters={
        'model__batch_size': [64],
        'model__epochs': [1, 2, 4],
        'model__num_dense_layer': [0, 1, 2],
        'model__loss': ['mean_squared_error'],
    },
    grid_parameters=evaluations.grid_parameters(),
)

result_handlers = [
    result_handlers.print_gridsearch_results,
]
