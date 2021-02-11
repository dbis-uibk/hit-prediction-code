"""Wide and deep model evaluation plan using all features."""
import os.path

from dbispipeline.evaluators import CvEpochEvaluator
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import hit_prediction_code.common as common
from hit_prediction_code.dataloaders import EssentiaLoader
import hit_prediction_code.evaluations as evaluations
from hit_prediction_code.models.ordinal import OrdinalClassifier
from hit_prediction_code.result_handlers import print_results_as_json
from hit_prediction_code.transformers.label import \
    convert_array_to_closest_labels

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

labels = common.weeks_labels()

pipeline = Pipeline([
    ('scale', MinMaxScaler()),
    ('model',
     OrdinalClassifier(
         ExtraTreesClassifier(),
         lambda y: convert_array_to_closest_labels(y, labels=labels),
     )),
])

evaluator = CvEpochEvaluator(
    cv=evaluations.cv(),
    scoring=evaluations.metrics.scoring(
        hit_nonhit_accuracy_score=None,
        categories=labels,
    ),
    scoring_step_size=1,
)

result_handlers = [
    print_results_as_json,
]
