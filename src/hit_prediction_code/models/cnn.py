# -*- coding: utf-8 -*-
"""CRNN model for hit song prediction."""
import logging

from sklearn.base import BaseEstimator, RegressorMixin
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import ELU
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.models import Model

from .wide_and_deep import dense_layers
from ..common import cached_model_predict
from ..common import cached_model_predict_clear

LOGGER = logging.getLogger(__name__)


def input_padding_layer(self, melgram_input, input_shape):
    # Input block
    padding = self.network_input_width - input_shape[1]
    left_pad = int(padding / 2)
    if padding % 2:
        right_pad = left_pad + 1
    else:
        right_pad = left_pad
    input_padding = ((0, 0), (left_pad, right_pad))
    hidden = ZeroPadding2D(padding=input_padding)(melgram_input)

    return hidden


def mel_cnn_layers(self, hidden):
    channel_axis = 3

    # Conv block 1
    hidden = Conv2D(self.layer_sizes['conv1'], (3, 3),
                    padding=self.padding,
                    name='conv1')(hidden)
    hidden = BatchNormalization(axis=channel_axis, name='bn1')(hidden)
    hidden = ELU()(hidden)
    hidden = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                          name='pool1')(hidden)
    hidden = Dropout(0.1, name='dropout1')(hidden)

    # Conv block 2
    hidden = Conv2D(self.layer_sizes['conv2'], (3, 3),
                    padding=self.padding,
                    name='conv2')(hidden)
    hidden = BatchNormalization(axis=channel_axis, name='bn2')(hidden)
    hidden = ELU()(hidden)
    hidden = MaxPooling2D(pool_size=(3, 3), strides=(3, 3),
                          name='pool2')(hidden)
    hidden = Dropout(0.1, name='dropout2')(hidden)

    # Conv block 3
    hidden = Conv2D(self.layer_sizes['conv3'], (3, 3),
                    padding=self.padding,
                    name='conv3')(hidden)
    hidden = BatchNormalization(axis=channel_axis, name='bn3')(hidden)
    hidden = ELU()(hidden)
    hidden = MaxPooling2D(pool_size=(4, 4), strides=(4, 4),
                          name='pool3')(hidden)
    hidden = Dropout(0.1, name='dropout3')(hidden)

    # Conv block 4
    hidden = Conv2D(self.layer_sizes['conv4'], (3, 3),
                    padding=self.padding,
                    name='conv4')(hidden)
    hidden = BatchNormalization(axis=channel_axis, name='bn4')(hidden)
    hidden = ELU()(hidden)
    hidden = MaxPooling2D(pool_size=(4, 4), strides=(4, 4),
                          name='pool4')(hidden)
    hidden = Dropout(0.1, name='dropout4')(hidden)

    return hidden


class CNNModel(BaseEstimator, RegressorMixin):

    def __init__(self,
                 layer_sizes=None,
                 batch_size=64,
                 epochs=100,
                 padding='same',
                 attention=False,
                 batch_normalization=False,
                 dropout_rate=None,
                 num_dense_layer=0,
                 dense_activation='relu',
                 output_activation=None,
                 loss='mean_absolute_error'):
        if layer_sizes is None:
            layer_sizes = {
                'conv1': 30,
                'conv2': 60,
                'conv3': 60,
                'conv4': 60,
                'cnn': 30,
                'dense': 30,
            }
        self.layer_sizes = layer_sizes
        self.batch_size = batch_size
        self.epochs = epochs
        self.padding = padding
        self.attention = attention
        self.batch_normalization = batch_normalization
        self.dropout_rate = dropout_rate
        self.num_dense_layer = num_dense_layer
        self.dense_activation = dense_activation
        self.output_activation = output_activation
        self.loss = loss

        self.network_input_width = 1200
        self.model = None

    def fit(self, data, labels):
        data = self._reshape_data(data)
        input_shape, output_shape = self._data_shapes(data, labels)
        self._create_model(input_shape, output_shape)

        self.model.fit(
            data,
            labels,
            batch_size=self.batch_size,
            epochs=self.epochs,
        )
        cached_model_predict_clear()

    def _data_shapes(self, data, labels):
        if data.shape[2] > self.network_input_width:
            raise ValueError('window_size > ' + str(self.network_input_width))
        input_shape = (data.shape[1], data.shape[2], data.shape[3])
        output_shape = labels.shape[1]

        return input_shape, output_shape

    def _create_model(self, input_shape, output_shape):
        melgram_input, output = self._crnn_layers(input_shape, output_shape)
        self.model = Model(inputs=melgram_input, outputs=output)
        self.model.compile(
            optimizer="adam",
            loss=self.loss,
            metrics=['mean_absolute_error'],
        )
        self.model.summary()

    def _crnn_layers(self, input_shape, output_shape):
        melgram_input = Input(shape=input_shape, dtype="float32")

        hidden = input_padding_layer(self, melgram_input, input_shape)
        hidden = mel_cnn_layers(self, hidden)

        # reshaping
        hidden = Reshape((12, self.layer_sizes['conv4']))(hidden)

        # reduce size
        hidden = MaxPooling1D(12)(hidden)
        hidden = Conv1D(self.layer_sizes['cnn'], 1)(hidden)
        hidden = Flatten()(hidden)

        dense_size = self.layer_sizes.setdefault(
            'dense',
            self.layer_sizes['cnn'],
        )

        dense_layer = dense_layers(self, dense_size, hidden)

        output = Dense(output_shape,
                       activation=self.output_activation,
                       name='output')(dense_layer)

        return melgram_input, output

    def predict(self, data):
        data = self._reshape_data(data)
        return cached_model_predict(self.model, data)

    def _reshape_data(self, data):
        data_shape = (*data.shape, 1)
        return data.reshape(data_shape)
