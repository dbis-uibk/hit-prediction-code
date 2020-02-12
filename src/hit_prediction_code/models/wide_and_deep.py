import numpy as np
from tensorflow.keras.layers import (Activation, BatchNormalization,
                                     Concatenate, Dense, Dropout, Input)

from tensorflow.keras.models import Model
from sklearn.base import BaseEstimator, RegressorMixin

from ..common import feature_columns


def dense_layers(self, dense_size, dense_layer):
    if self.batch_normalization:
        use_bias = False
        activation = None
    else:
        use_bias = True
        activation = self.dense_activation

    for i in range(1, self.num_dense_layer + 1):
        dense_layer = Dense(dense_size,
                            activation=activation,
                            name='dense-' + str(i),
                            use_bias=use_bias)(dense_layer)
        if self.batch_normalization:
            dense_layer = BatchNormalization(name='bn-' + str(i))(dense_layer)
            dense_layer = Activation(self.dense_activation,
                                     name='activation-' + str(i))(dense_layer)
        if self.dropout_rate:
            dense_layer = Dropout(self.dropout_rate,
                                  name='dropout-' + str(i))(dense_layer)

    return dense_layer


class WideAndDeep(BaseEstimator, RegressorMixin):

    def __init__(self,
                 loss='mse',
                 optimizer='adam',
                 metrics=['mae'],
                 deep_activation='sigmoid',
                 dense_activation='relu',
                 output_activation=None,
                 epochs=1,
                 batch_size=None,
                 features=None,
                 batch_normalization=False,
                 dropout_rate=None,
                 dense_output_size=None,
                 num_dense_layer=2,
                 **kwargs):
        self.input_list = []
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.deep_activation = deep_activation
        self.dense_activation = dense_activation
        self.output_activation = output_activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.features = features
        self.batch_normalization = batch_normalization
        self.dropout_rate = dropout_rate
        self.dense_output_size = dense_output_size
        self.num_dense_layer = num_dense_layer
        self._config = {
            'loss': loss,
            'optimizer': optimizer,
            'metrics': metrics,
            'deep_activation': deep_activation,
            'dense_activation': dense_activation,
            'output_activation': output_activation,
            'epochs': epochs,
            'batch_size': batch_size,
            'features': features,
            'batch_normalization': batch_normalization,
            'dropout_rate': dropout_rate,
            'dense_output_size': dense_output_size,
            'num_dense_layer': num_dense_layer,
            **kwargs,
        }
        self._model = None

    @property
    def configuration(self):
        return self._config

    def _build_model(self):
        input_list = []
        input_type_list = []
        concat_list = []
        for i, feature in enumerate(self.features):
            split, part = feature
            if part is None:
                part = 'deep'
            name = str(part) + '-input-' + str(i)
            input_list.append(Input(shape=(len(split),), name=name))
            input_type_list.append(part)

        for i, input_entry in enumerate(zip(input_list, input_type_list)):
            entry, part = input_entry
            if part == 'wide':
                concat_list.append(entry)
            else:
                name = 'fab-' + str(i)
                concat_list.append(
                    Dense(1, activation=self.deep_activation,
                          name=name)(entry))

        concat_tensor = Concatenate(axis=-1,
                                    name='concat_wide_and_deep')(concat_list)

        if self.dense_output_size:
            dense_output_size = self.dense_output_size
        else:
            dense_output_size = len(input_list)

        dense_layer = dense_layers(self, dense_output_size, concat_tensor)

        use_bias = not self.batch_normalization
        output = Dense(1,
                       activation=self.output_activation,
                       name='output',
                       use_bias=use_bias)(dense_layer)

        model = Model(inputs=input_list, outputs=output)
        model.compile(loss=self.loss,
                      optimizer=self.optimizer,
                      metrics=self.metrics)

        return model

    def _split_features(self, x):
        features = []
        for index, _ in self.features:
            feature = x[..., index]
            features.append(feature)
        return features

    def fit(self, x, y=None):
        if self._model:
            raise NotImplementedError('Refitting the model is not implemented')

        features = self._split_features(x)
        self._model = self._build_model()
        self._model.summary()
        self._model.fit(x=features,
                        y=y,
                        batch_size=self.batch_size,
                        epochs=self.epochs)

    def predict(self, x):
        features = self._split_features(x)
        return self._model.predict(features)
