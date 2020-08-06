"""Wide and deep model evaluation plan using all features."""
import os.path

from dbispipeline.evaluators import CvEpochEvaluator
import dbispipeline.result_handlers
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import hit_prediction_code.common as common
from hit_prediction_code.dataloaders import EssentiaLoader
import hit_prediction_code.evaluations as evaluations
from hit_prediction_code.models.simple_nn import SimpleNN

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
    nan_value=0,
)

pipeline = Pipeline([
    ('scale', MinMaxScaler()),
    ('model',
     SimpleNN(
         epochs=1000,
         dense_sizes=(
             256,
             128,
             64,
             32,
             16,
             8,
             4,
             2,
         ),
     )),
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
        )),
    scoring_step_size=10,
)

result_handlers = [
    dbispipeline.result_handlers.print_gridsearch_results,
]
