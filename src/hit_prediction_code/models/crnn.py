# -*- coding: utf-8 -*-
"""CRNN model for hit song prediction."""
import logging

import tensorflow.compat.v2.keras.backend as keras_backend
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Multiply
from tensorflow.keras.layers import Permute
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Model

from .building_blocks import HitPredictionModel
from .building_blocks import dense_layers
from .building_blocks import input_padding_layer
from .building_blocks import mel_cnn_layers

LOGGER = logging.getLogger(__name__)


class CRNNModel(HitPredictionModel):
    """CRNN Model designed for hit song prediction."""

    def __init__(self,
                 layer_sizes=None,
                 batch_size=64,
                 epochs=100,
                 padding='same',
                 attention=False,
                 batch_normalization=False,
                 cnn_activation='elu',
                 cnn_batch_normalization=True,
                 dropout_rate=None,
                 num_dense_layer=0,
                 dense_activation='relu',
                 output_activation=None,
                 loss='mean_absolute_error'):
        """Initializes the CRNN Model object.

        Args:
            layer_sizes: a dict containing the layer sizes (width) of the CRNN.
            batch_size: the batch size used to train the model.
            epochs: the number of epochs used during training.
            padding: the padding type used for inputs.
            attention: specifies if an attention mechanism is used between the
                dense layers and the RNN layers of the CRNN model.
            batch_normalization: configures if batch normalization is used for
                the dense network part.
            cnn_activation: the activation function used in the cnn blocks.
            cnn_batch_normalization: configures if batch normalization is used
                for the cnn part of the network.
            dropout_rate: the dropout rate used for the dense part.
            num_dense_layer: the number of dense layers in the dense part.
            dense_activation: the activation function used for the dense part.
            output_activation: the activation function used for the output.
            loss: the loss function used to train the network.

        """
        super().__init__(
            metrics=['mean_absolute_error'],
            optimizer='adam',
        )

        if layer_sizes is None:
            layer_sizes = {
                'conv1': 30,
                'conv2': 60,
                'conv3': 60,
                'conv4': 60,
                'rnn': 30,
                'dense': 30,
            }
        self.layer_sizes = layer_sizes
        self.batch_size = batch_size
        self.epochs = epochs
        self.padding = padding
        self.attention = attention
        self.batch_normalization = batch_normalization
        self.cnn_activation = cnn_activation
        self.cnn_batch_normalization = cnn_batch_normalization
        self.dropout_rate = dropout_rate
        self.num_dense_layer = num_dense_layer
        self.dense_activation = dense_activation
        self.output_activation = output_activation
        self.loss = loss

        self.network_input_width = 1200
        self.model = None

    @property
    def cnn_activation(self):
        """Property specifying the activation used for the cnn blocks."""
        return self._config.get('cnn_activation')

    @cnn_activation.setter
    def cnn_activation(self, value):
        self._config['cnn_activation'] = value

    @property
    def cnn_batch_normalization(self):
        """Property specifying if cnn part uses batch normalization."""
        return self._config.get('cnn_batch_normalization')

    @cnn_batch_normalization.setter
    def cnn_batch_normalization(self, value):
        self._config['cnn_batch_normalization'] = value

    def _data_shapes(self, data, labels):
        if data.shape[2] > self.network_input_width:
            raise ValueError('window_size > ' + str(self.network_input_width))
        input_shape = (data.shape[1], data.shape[2], data.shape[3])
        output_shape = labels.shape[1]

        return input_shape, output_shape

    def _create_model(self, input_shape, output_shape):
        melgram_input = Input(shape=input_shape, dtype='float32')

        hidden = input_padding_layer(
            self.network_input_width,
            melgram_input,
        )
        hidden = mel_cnn_layers(
            self.layer_sizes,
            self.padding,
            hidden,
            batch_normalization=self.cnn_batch_normalization,
            activation=self.cnn_activation,
        )

        # reshaping
        hidden = Reshape((12, self.layer_sizes['conv4']))(hidden)

        # GRU block 1, 2, output
        hidden = GRU(self.layer_sizes['rnn'],
                     return_sequences=True,
                     name='gru1')(hidden)
        hidden = GRU(self.layer_sizes['rnn'],
                     return_sequences=self.attention,
                     name='gru2')(hidden)

        if self.attention:
            attention = Dense(1)(hidden)
            attention = Flatten()(attention)
            attention_act = Activation('softmax')(attention)
            attention = RepeatVector(self.layer_sizes['rnn'])(attention_act)
            attention = Permute((2, 1))(attention)

            merged = Multiply()([hidden, attention])
            hidden = Lambda(lambda xin: keras_backend.sum(xin, axis=1))(merged)

        dense_size = self.layer_sizes.setdefault(
            'dense',
            self.layer_sizes['rnn'],
        )

        dense_layer = dense_layers(
            self.batch_normalization,
            self.dropout_rate,
            dense_size,
            self.num_dense_layer,
            self.dense_activation,
            hidden,
        )

        output = Dense(output_shape,
                       activation=self.output_activation,
                       name='output')(dense_layer)

        self.model = Model(inputs=melgram_input, outputs=output)

    def _reshape_data(self, data):
        data_shape = (*data.shape, 1)
        return data.reshape(data_shape)
