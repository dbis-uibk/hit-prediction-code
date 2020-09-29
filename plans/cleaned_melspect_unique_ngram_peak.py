"""Wide and deep model evaluation plan using all features."""
import os.path

from dbispipeline.evaluators import GridEvaluator
import dbispipeline.result_handlers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline

from hit_prediction_code.dataloaders import MelSpectLoader
import hit_prediction_code.evaluations as evaluations
from hit_prediction_code.transformers.mel_spect import MelSpectScaler
from hit_prediction_code.transformers.mel_spect import ZOrderTransformer

PATH_PREFIX = 'data/hit_song_prediction_ismir2020/processed'

dataloader = MelSpectLoader(
    dataset_path=os.path.join(
        PATH_PREFIX,
        'msd_bb_mbid_cleaned_matches_melspect_unique.pickle.xz',
    ),
    features='librosa_melspectrogram',
    label='peakPos',
    nan_value=150,
)

pipeline = Pipeline([
    ('scale', MelSpectScaler()),
    ('z-order', ZOrderTransformer()),
    ('vectorizer', CountVectorizer(analyzer='word')),
    ('tf-idf', TfidfTransformer()),
    ('model', SGDRegressor()),
])

evaluator = GridEvaluator(
    parameters={
        'vectorizer__max_df': (0.5, 0.75, 1.0),
        # 'vectorizer__max_features': (None, 5000, 10000, 50000),
        'vectorizer__ngram_range': (
            (1, 1), (1, 2), (1, 3), (1, 4), (1, 8)),  # unigrams or bigrams
        # 'tf-idf__use_idf': (True, False),
        # 'tf-idf__norm': ('l1', 'l2'),
        'model__max_iter': (20,),
        # 'model__alpha': (0.00001, 0.000001),
        # 'model__penalty': ('l2', 'elasticnet'),
        # 'model__max_iter': (10, 50, 80),
    },
    grid_parameters=evaluations.grid_parameters(),
)

result_handlers = [
    dbispipeline.result_handlers.print_gridsearch_results,
]
