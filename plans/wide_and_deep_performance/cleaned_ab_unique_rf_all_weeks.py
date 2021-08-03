"""Random forest model evaluation plan using all features."""
import os.path

from dbispipeline.evaluators import CvEpochEvaluator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import hit_prediction_code.common as common
from hit_prediction_code.dataloaders import EssentiaLoader
import hit_prediction_code.evaluations as evaluations
from hit_prediction_code.models.tree import RandomForestRegressor
from hit_prediction_code.result_handlers import print_results_as_json

PATH_PREFIX = 'data/hit_song_prediction_ismir2020/processed'

dataloader = EssentiaLoader(
    dataset_path=os.path.join(
        PATH_PREFIX,
        'msd_bb_mbid_cleaned_matches_ab_unique.parquet',
    ),
    features=[
        *common.all_no_year_list(),
    ],
    label='weeks',
    nan_value=common.weeks_non_hit_value(),
)

pipeline = Pipeline([
    ('scale', MinMaxScaler()),
    ('model', RandomForestRegressor()),
])

evaluator = CvEpochEvaluator(
    cv=evaluations.cv(),
    scoring=evaluations.metrics.scoring(
        hit_nonhit_accuracy_score=lambda evaluator, x, y: evaluations.metrics.
        hit_nonhit_accuracy_score(
            evaluator,
            x,
            y,
            threshold=0.5,
        ),
        categories=common.weeks_labels(),
    ),
    scoring_step_size=1,
)

result_handlers = [
    print_results_as_json,
]
