"""CNN model evaluation plan."""
from dbispipeline.evaluators import GridEvaluator
import dbispipeline.result_handlers as result_handlers
from sklearn.pipeline import Pipeline

from hit_prediction_code.dataloaders import MelSpectLoader
import hit_prediction_code.evaluations as evaluations
from hit_prediction_code.models.cnn import CNNModel

dataloader = MelSpectLoader(
    dataset_path='data/hit_song_prediction_ismir2020/processed/msd_bb_balanced_dev_sample.pickle',
    features='librosa_melspectrogram',
    label='peak',
    nan_value=150,
)

pipeline = Pipeline([
    (
        'model',
        CNNModel(
            layer_sizes={
                'conv1': 30,
                'conv2': 60,
                'conv3': 60,
                'conv4': 60,
                'cnn': 30,
                'dense': 30,
            }),
    ),
])

evaluator = GridEvaluator(
    parameters={
        'model__batch_size': [64],
        'model__epochs': [1, 2],
        'model__num_dense_layer': [2],
        'model__batch_normalization': [False],
        'model__cnn_batch_normalization': [False],
        'model__dense_activation': ['elu'],
        'model__output_activation': ['elu'],
        'model__dropout_rate': [0.1],
        'model__loss': ['mean_squared_error'],
    },
    grid_parameters=evaluations.grid_parameters(),
)

result_handlers = [
    result_handlers.print_gridsearch_results,
]