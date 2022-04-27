"""Plan using all features."""
import os.path

from dbispipeline.evaluators import CvEpochEvaluator
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import hit_prediction_code.common as common
from hit_prediction_code.dataloaders import ClassLoaderWrapper
from hit_prediction_code.dataloaders import EssentiaLoader
from hit_prediction_code.dataloaders import QcutLoaderWrapper
import hit_prediction_code.evaluations as evaluations
from hit_prediction_code.models.pairwise import PairwiseOrdinalModel
from hit_prediction_code.result_handlers import print_results_as_json
from hit_prediction_code.transformers.label import compute_hit_score_on_df

PATH_PREFIX = 'data/hit_song_prediction_msd_bb_lfm_ab/processed'

number_of_classes = 55

dataloader = ClassLoaderWrapper(
    wrapped_loader=QcutLoaderWrapper(
        wrapped_loader=EssentiaLoader(
            dataset_path=os.path.join(
                PATH_PREFIX,
                'hsp-s_acousticbrainz.parquet',
            ),
            features=[
                *common.all_no_year_list(),
            ],
            label='yang_hit_score',
            nan_value=0,
            data_modifier=lambda df: compute_hit_score_on_df(
                df,
                pc_column='lastfm_playcount',
                lc_column='lastfm_listener_count',
                hit_score_column='yang_hit_score',
            ),
        ),
        number_of_bins=number_of_classes,
    ),
    labels=list(range(number_of_classes)),
)

pipeline = Pipeline([
    ('scale', MinMaxScaler()),
    ('model',
     PairwiseOrdinalModel(
         wrapped_model=LinearRegression(),
         pairs_factor=3.,
         threshold_type='average',
         pair_strategy='random',
         pair_encoding='concat',
         threshold_sample_training=False,
     )),
])

evaluator = CvEpochEvaluator(
    cv=evaluations.cv(),
    scoring=evaluations.metrics.ordinal_classifier_scoring(),
    scoring_step_size=1,
)

result_handlers = [
    print_results_as_json,
]
