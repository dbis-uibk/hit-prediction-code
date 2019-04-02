from keras.layers import Concatenate, Dense, Input
from keras.models import Model

from sklearn.base import BaseEstimator, RegressorMixin

from common import feature_columns
import numpy as np


class WideAndDeep(BaseEstimator, RegressorMixin):
    def __init__(self, features=None, **kwargs):
        self.input_list = []
        self.features = features
        self._config = kwargs
        self._model = None

    @property
    def configuration(self):
        return self._config

    def _build_model(self):
        input_list = []
        concat_list = []
        for split in self.features:
            input_list.append(Input(shape=(len(split),)))

        for entry in input_list:
            activation = 'sigmoid'
            concat_list.append(Dense(1, activation=activation)(entry))

        concat_tensor = Concatenate(axis=-1)(concat_list)

        activation = 'relu'
        dense_layer = Dense(256, activation=activation)(concat_tensor)
        dense_layer = Dense(256, activation=activation)(dense_layer)
        output = Dense(1,activation='sigmoid')(dense_layer)

        model = Model(inputs=input_list, outputs=output)
        model.compile(loss='mse', optimizer='adam', metrics=['mae'])

        return model

    def _split_features(self, x):
        features = []
        for index in self.features:
            feature = x[...,index]
            features.append(feature)
        return features

    def fit(self, x, y=None):
        if self._model:
            raise NotImplementedError('Refitting the model is not implemented')

        features = self._split_features(x)
        self._model = self._build_model()
        self._model.fit(features, y)

    def predict(self, x):
        features = self._split_features(x)
        return self._model.predict(features)
