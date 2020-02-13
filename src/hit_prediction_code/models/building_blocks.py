# -*- coding: utf-8 -*-
"""Building blocks that can be reused in the models."""
from abc import ABCMeta
from abc import abstractmethod
import logging

from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import ELU
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D

from ..common import cached_model_predict
from ..common import cached_model_predict_clear

LOGGER = logging.getLogger(__name__)


def input_padding_layer(network_input_width, melgram_input, input_shape):
    # Input block
    padding = network_input_width - input_shape[1]
    left_pad = int(padding / 2)
    if padding % 2:
        right_pad = left_pad + 1
    else:
        right_pad = left_pad
    input_padding = ((0, 0), (left_pad, right_pad))
    hidden = ZeroPadding2D(padding=input_padding)(melgram_input)

    return hidden


def mel_cnn_layers(layer_sizes, padding, hidden):
    channel_axis = 3

    # Conv block 1
    hidden = Conv2D(layer_sizes['conv1'], (3, 3),
                    padding=padding,
                    name='conv1')(hidden)
    hidden = BatchNormalization(axis=channel_axis, name='bn1')(hidden)
    hidden = ELU()(hidden)
    hidden = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                          name='pool1')(hidden)
    hidden = Dropout(0.1, name='dropout1')(hidden)

    # Conv block 2
    hidden = Conv2D(layer_sizes['conv2'], (3, 3),
                    padding=padding,
                    name='conv2')(hidden)
    hidden = BatchNormalization(axis=channel_axis, name='bn2')(hidden)
    hidden = ELU()(hidden)
    hidden = MaxPooling2D(pool_size=(3, 3), strides=(3, 3),
                          name='pool2')(hidden)
    hidden = Dropout(0.1, name='dropout2')(hidden)

    # Conv block 3
    hidden = Conv2D(layer_sizes['conv3'], (3, 3),
                    padding=padding,
                    name='conv3')(hidden)
    hidden = BatchNormalization(axis=channel_axis, name='bn3')(hidden)
    hidden = ELU()(hidden)
    hidden = MaxPooling2D(pool_size=(4, 4), strides=(4, 4),
                          name='pool3')(hidden)
    hidden = Dropout(0.1, name='dropout3')(hidden)

    # Conv block 4
    hidden = Conv2D(layer_sizes['conv4'], (3, 3),
                    padding=padding,
                    name='conv4')(hidden)
    hidden = BatchNormalization(axis=channel_axis, name='bn4')(hidden)
    hidden = ELU()(hidden)
    hidden = MaxPooling2D(pool_size=(4, 4), strides=(4, 4),
                          name='pool4')(hidden)
    hidden = Dropout(0.1, name='dropout4')(hidden)

    return hidden


def dense_layers(batch_normalization, dropout_rate, dense_size,
                 num_dense_layer, dense_activation, dense_layer):
    if batch_normalization:
        use_bias = False
        activation = None
    else:
        use_bias = True
        activation = dense_activation

    for i in range(1, num_dense_layer + 1):
        dense_layer = Dense(dense_size,
                            activation=activation,
                            name='dense-' + str(i),
                            use_bias=use_bias)(dense_layer)
        if batch_normalization:
            dense_layer = BatchNormalization(name='bn-' + str(i))(dense_layer)
            dense_layer = Activation(dense_activation,
                                     name='activation-' + str(i))(dense_layer)
        if dropout_rate:
            dense_layer = Dropout(dropout_rate,
                                  name='dropout-' + str(i))(dense_layer)

    return dense_layer


class HitPredictionModel(BaseEstimator, RegressorMixin, metaclass=ABCMeta):

    def __init__(self, **kwargs):
        self._config = {
            **kwargs,
        }
        self._model = None

    @property
    def batch_normalization(self):
        return self._config.get('batch_normalization')

    @batch_normalization.setter
    def batch_normalization(self, value):
        self._config['batch_normalization'] = value

    @property
    def batch_size(self):
        return self._config.get('batch_size')

    @batch_size.setter
    def batch_size(self, value):
        self._config['batch_size'] = value

    @property
    def configuration(self):
        return self._config

    @property
    def dense_activation(self):
        return self._config.get('dense_activation')

    @dense_activation.setter
    def dense_activation(self, value):
        self._config['dense_activation'] = value

    @property
    def dropout_rate(self):
        return self._config.get('dropout_rate')

    @dropout_rate.setter
    def dropout_rate(self, value):
        self._config['dropout_rate'] = value

    @property
    def epochs(self):
        return self._config.get('epochs')

    @epochs.setter
    def epochs(self, value):
        self._config['epochs'] = value

    @property
    def loss(self):
        return self._config.get('loss')

    @loss.setter
    def loss(self, value):
        self._config['loss'] = value

    @property
    def metrics(self):
        return self._config['metrics']

    @metrics.setter
    def metrics(self, value):
        self._config['metrics'] = value

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def num_dense_layer(self):
        return self._config.get('num_dense_layer')

    @num_dense_layer.setter
    def num_dense_layer(self, value):
        self._config['num_dense_layer'] = value

    @property
    def optimizer(self):
        return self._config.get('optimizer')

    @optimizer.setter
    def optimizer(self, value):
        self._config['optimizer'] = value

    @property
    def output_activation(self):
        return self._config.get('output_activation')

    @output_activation.setter
    def output_activation(self, value):
        self._config['output_activation'] = value

    def fit(self, data, labels):
        if self.model is not None:
            raise NotImplementedError('Refitting the model is not implemented')

        data = self._reshape_data(data)
        input_shape, output_shape = self._data_shapes(data, labels)
        self._create_model(input_shape, output_shape)

        self.model.compile(
            loss=self.loss,
            metrics=self.metrics,
            optimizer=self.optimizer,
        )
        self.model.summary()

        self.model.fit(
            x=data,
            y=labels,
            batch_size=self.batch_size,
            epochs=self.epochs,
        )
        cached_model_predict_clear()

    def predict(self, data):
        data = self._reshape_data(data)
        return cached_model_predict(self.model, data)

    @abstractmethod
    def _create_model(self, input_shape, output_shape):
        self.model = None

    @abstractmethod
    def _data_shapes(self, data, labels):
        pass

    @abstractmethod
    def _reshape_data(self, data):
        pass
