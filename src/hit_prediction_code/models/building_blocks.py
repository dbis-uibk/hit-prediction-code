# -*- coding: utf-8 -*-
"""Building blocks that can be reused in the models."""
import logging

from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import ELU
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D

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
