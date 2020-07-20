"""CNN model evaluation plan."""
from dbispipeline.evaluators import CvEpochEvaluator
import dbispipeline.result_handlers as result_handlers
from sklearn.pipeline import Pipeline

from hit_prediction_code.dataloaders import MelSpectLoader
import hit_prediction_code.evaluations as evaluations
from hit_prediction_code.models.cnn import CNNModel

dataloader = MelSpectLoader(
    dataset_path='data/hit_song_prediction_ismir2020/processed/msd_bb_balanced.pickle',
    features='librosa_melspectrogram',
    label='peak',
    nan_value=150,
)

pipeline = Pipeline([
    (
        'model',
        CNNModel(
            batch_size=64,
            epochs=300,
            num_dense_layer=2,
            batch_normalization=False,
            cnn_activation='selu',
            cnn_batch_normalization=False,
            dense_activation='selu',
            output_activation='elu',
            dropout_rate=0.1,
            loss='mean_squared_error',
            layer_sizes={
                'conv1': 30,
                'conv2': 60,
                'conv3': 60,
                'conv4': 60,
                'cnn': 30,
                'dense': 30,
            },
            input_normalization=True,
        ),
    ),
])

evaluator = CvEpochEvaluator(
    cv=evaluations.cv(),
    scoring=evaluations.metrics.scoring(),
    scoring_step_size=10,
)

result_handlers = [
    result_handlers.print_gridsearch_results,
]
