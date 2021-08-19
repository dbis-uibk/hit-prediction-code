"""LogisticRegressionClassifier plan using mean and std mel-spect features."""
import os.path

from dbispipeline.evaluators import CvEpochEvaluator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from hit_prediction_code.dataloaders import BinaryClassLoaderWrapper
from hit_prediction_code.dataloaders import MelSpectMeanStdLoader
import hit_prediction_code.evaluations as evaluations
from hit_prediction_code.models.linear import LogisticRegressionClassifier
from hit_prediction_code.result_handlers import print_results_as_json
from hit_prediction_code.transformers.label import compute_hit_score_on_df

PATH_PREFIX = 'data/hit_song_prediction_ismir2020/processed'

dataloader = BinaryClassLoaderWrapper(wrapped_loader=MelSpectMeanStdLoader(
    dataset_path=os.path.join(
        PATH_PREFIX,
        'msd_bb_mbid_cleaned_matches_melspect_db_unique.pickle',
    ),
    features='librosa_melspectrogram',
    label='yang_hit_score',
    nan_value=0,
    data_modifier=lambda df: compute_hit_score_on_df(
        df,
        pc_column='lastfm_playcount',
        lc_column='lastfm_listener_count',
        hit_score_column='yang_hit_score',
    ),
))

pipeline = Pipeline([
    ('scale', MinMaxScaler()),
    ('model', LogisticRegressionClassifier()),
])

evaluator = CvEpochEvaluator(
    cv=evaluations.cv(),
    scoring=evaluations.metrics.classifier_scoring(),
    scoring_step_size=1,
)

result_handlers = [
    print_results_as_json,
]
