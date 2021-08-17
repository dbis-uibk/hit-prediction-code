"""LinearRegression model plan using mean and std mel-spect features."""
import os.path

from dbispipeline.evaluators import CvEpochEvaluator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from hit_prediction_code.dataloaders import MelSpectLoader
import hit_prediction_code.evaluations as evaluations
from hit_prediction_code.models.cnn import FCN
from hit_prediction_code.result_handlers import print_results_as_json
from hit_prediction_code.transformers.label import compute_hit_score_on_df

PATH_PREFIX = 'data/hit_song_prediction_ismir2020/processed'

dataloader = MelSpectLoader(
    dataset_path=os.path.join(
        PATH_PREFIX,
        'msd_bb_mbid_cleaned_matches_melspect_unique.pickle',
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
)

pipeline = Pipeline([
    ('scale', MinMaxScaler()),
    ('model', FCN(epochs=100)),
])

evaluator = CvEpochEvaluator(
    cv=evaluations.cv(),
    scoring=evaluations.metrics.scoring(
        hit_nonhit_accuracy_score=None,
        categories=None,
    ),
    scoring_step_size=evaluations.scoring_step_size(),
)

result_handlers = [
    print_results_as_json,
]
