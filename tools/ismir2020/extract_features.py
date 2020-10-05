"""Extracts features from melspects."""
import os.path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

from hit_prediction_code.dataloaders import MelSpectLoader
from hit_prediction_code.transformers.mel_spect import FloatListToSentence
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
    ('sentence', FloatListToSentence(round_decimals=3)),
    ('vector', CountVectorizer(analyzer='word')),
    ('tf-idf', TfidfTransformer()),
])

tf_idf = pd.DataFrame.sparse.from_spmatrix(
    pipeline.fit_transform(dataloader.load()[0]))
tf_idf.to_pickle(
    os.path.join(
        PATH_PREFIX,
        'analytics_tfidf_msd_bb_mbid_cleaned_matches_melspect_unique.pickle.xz',  # noqa:E501
    ),
    'xz',
)
