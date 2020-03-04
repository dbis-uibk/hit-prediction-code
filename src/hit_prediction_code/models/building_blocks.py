# -*- coding: utf-8 -*-
"""Building blocks that can be reused in the models."""
from abc import ABCMeta
from abc import abstractmethod
import logging

from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import AlphaDropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D

from ..common import cached_model_predict
from ..common import cached_model_predict_clear

LOGGER = logging.getLogger(__name__)


def input_padding_layer(network_input_width, input_layer):
    """Adds padding to the input to ensure that the input width fits.

    Adds padding left and right of the input to enlarge the input to fit the
    expected network width.

    Args:
        network_input_width: the expected input width of the network. The input
            is padded to have that size.
        input_layer: the input layer that is padded.

    Returns: the resulting layer including the padding.

    """
    padding = network_input_width - int(input_layer.shape[2])
    left_pad = int(padding / 2)
    if padding % 2:
        right_pad = left_pad + 1
    else:
        right_pad = left_pad
    input_padding = ((0, 0), (left_pad, right_pad))
    hidden = ZeroPadding2D(padding=input_padding)(input_layer)

    return hidden


def _add_conv_dropout_block(config, hidden):
    channel_axis = 3

    hidden = Conv2D(filters=config['filter_size'],
                    kernel_size=config['kernel_size'],
                    padding=config['padding'],
                    name='conv' + config['name_suffix'])(hidden)
    if config['batch_normalization'] is True:
        hidden = BatchNormalization(axis=channel_axis, name='bn1')(hidden)
    hidden = Activation(config['activation'])(hidden)

    hidden = MaxPooling2D(pool_size=config['pool_size'],
                          strides=config['pool_stride'],
                          name='pool' + config['name_suffix'])(hidden)
    if config['activation'] == 'selu':
        hidden = AlphaDropout(
            config['dropout_rate'],
            name='alpha_dropout' + config['name_suffix'],
        )(hidden)
    else:
        hidden = Dropout(
            config['dropout_rate'],
            name='dropout' + config['name_suffix'],
        )(hidden)

    return hidden


def mel_cnn_layers(layer_sizes,
                   padding,
                   hidden,
                   batch_normalization=True,
                   activation='elu'):
    """Creates the CNN layers used to process the mel specs.

    This CNN builing block consists of four blocks each consisting of a 2D
    convolution layer followed by an optional batch normalization layer
    followed by an activation and a dropout layer. In case of SELU activation,
    an alpha dropout layer is used instead of the normal dropout layer.

    The convolution layer kernel sizes are (3,3), (3, 3), (3, 3), (3, 3)
    respectively. The max pool sizes and strides of the max pooling layers are
    (2, 2), (3, 3), (4, 4), (4, 4) respectively. Further, the dropout rate is
    set to 0.1

    Args:
        layer_sizes: a dict containing filter sizes used of the convolution
            layers.
        padding: the padding type used for the convolution blocks.
        hidden: the hidden layer where the CNN layers are are connected to.
        batch_normalization: specifies if the optinal batch normalization
            layers are used.
        activation: the name (as a string) of the used activation function.
    Returns: the last layer of that CNN layer block.

    """
    dropout_rate = 0.1

    kernel_size = {
        'conv1': (3, 3),
        'conv2': (3, 3),
        'conv3': (3, 3),
        'conv4': (3, 3),
    }

    pool_size = {
        'pool1': (2, 2),
        'pool2': (3, 3),
        'pool3': (4, 4),
        'pool4': (4, 4),
    }

    pool_strides = {
        'pool1': (2, 2),
        'pool2': (3, 3),
        'pool3': (4, 4),
        'pool4': (4, 4),
    }

    # create 4 conv blocks
    for block in range(1, 5):
        block = str(block)

        hidden = _add_conv_dropout_block(
            config={
                'name_suffix': block,
                'filter_size': layer_sizes['conv' + block],
                'kernel_size': kernel_size['conv' + block],
                'padding': padding,
                'batch_normalization': batch_normalization,
                'pool_size': pool_size['pool' + block],
                'pool_stride': pool_strides['pool' + block],
                'dropout_rate': dropout_rate,
                'activation': activation,
            },
            hidden=hidden,
        )

    return hidden


def dense_layers(batch_normalization, dropout_rate, dense_size,
                 num_dense_layer, dense_activation, hidden_layer):
    """Creates a dense layer block.

    Args:
        batch_normalization: specifies if batch normalization is used. If None,
            no batch normalization is used.
        dropout_rate: specifies the dropout_rate used. If None, no dropout
            layer is used.
        dense_size: the output size of each dense layer in that block of dense
            layers.
        num_dense_layer: the number of dense layers contained in that block.
        dense_activation: the activation used for the dense layers. If selu is
            used, alpha dropout is used instead of normal dropout.
        hidden_layer: the hidden layer passed to this dense layer block.

    Returns:
        The last dense layer in this block. Or the hidden_layer if
        num_dense_layer is 0.

    """
    if batch_normalization:
        use_bias = False
        activation = None
    else:
        use_bias = True
        activation = dense_activation

    for i in range(1, num_dense_layer + 1):
        hidden_layer = Dense(dense_size,
                             activation=activation,
                             name='dense-' + str(i),
                             use_bias=use_bias)(hidden_layer)
        if batch_normalization:
            hidden_layer = BatchNormalization(name='bn-' +
                                              str(i))(hidden_layer)
            hidden_layer = Activation(dense_activation,
                                      name='activation-' +
                                      str(i))(hidden_layer)
        if dropout_rate:
            if dense_activation == 'selu':
                hidden_layer = AlphaDropout(
                    dropout_rate,
                    name='alpha_dropout-' + str(i),
                )(hidden_layer)
            else:
                hidden_layer = Dropout(
                    dropout_rate,
                    name='dropout-' + str(i),
                )(hidden_layer)

    return hidden_layer


class HitPredictionModel(BaseEstimator, RegressorMixin, metaclass=ABCMeta):
    """Abstract base class to build hit prediction models."""

    def __init__(self, **kwargs):
        """Initializes the model.

        Args:
            kwargs: that are stored in the model configuration.
        """
        self._config = {
            **kwargs,
        }
        self._model = None

    @property
    def batch_normalization(self):
        """Property specifying if batch normalization is used."""
        return self._config.get('batch_normalization')

    @batch_normalization.setter
    def batch_normalization(self, value):
        self._config['batch_normalization'] = value

    @property
    def batch_size(self):
        """Property specifying the batch size used for training."""
        return self._config.get('batch_size')

    @batch_size.setter
    def batch_size(self, value):
        self._config['batch_size'] = value

    @property
    def configuration(self):
        """Property that contains the configuration of the model."""
        return self._config

    @property
    def dense_activation(self):
        """Property specifying which activation is used for dense layers."""
        return self._config.get('dense_activation')

    @dense_activation.setter
    def dense_activation(self, value):
        self._config['dense_activation'] = value

    @property
    def dropout_rate(self):
        """Property specifying the dropout rate that is used."""
        return self._config.get('dropout_rate')

    @dropout_rate.setter
    def dropout_rate(self, value):
        self._config['dropout_rate'] = value

    @property
    def epochs(self):
        """Property specifying the number of epochs used for training."""
        return self._config.get('epochs')

    @epochs.setter
    def epochs(self, value):
        self._config['epochs'] = value

    @property
    def loss(self):
        """Property specifying the loss function used for training."""
        return self._config.get('loss')

    @loss.setter
    def loss(self, value):
        self._config['loss'] = value

    @property
    def metrics(self):
        """Property specifying a list of metrics used to evaluate training."""
        return self._config['metrics']

    @metrics.setter
    def metrics(self, value):
        self._config['metrics'] = value

    @property
    def model(self):
        """Property holding the model."""
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def num_dense_layer(self):
        """Property specifying the number of dense layers."""
        return self._config.get('num_dense_layer')

    @num_dense_layer.setter
    def num_dense_layer(self, value):
        self._config['num_dense_layer'] = value

    @property
    def optimizer(self):
        """Property specifying the optimizer used for training."""
        return self._config.get('optimizer')

    @optimizer.setter
    def optimizer(self, value):
        self._config['optimizer'] = value

    @property
    def output_activation(self):
        """Property specifying the output activation that is used."""
        return self._config.get('output_activation')

    @output_activation.setter
    def output_activation(self, value):
        self._config['output_activation'] = value

    def fit(self, data, target):
        """Fits the model.

        Args:
            data: the training samples.
            target: the target values for the training samples.

        """
        if self.model is not None:
            raise NotImplementedError('Refitting the model is not implemented')

        data = self._reshape_data(data)
        input_shape, output_shape = self._data_shapes(data, target)
        self._create_model(input_shape, output_shape)

        self.model.compile(
            loss=self.loss,
            metrics=self.metrics,
            optimizer=self.optimizer,
        )
        self.model.summary()

        self.model.fit(
            x=data,
            y=target,
            batch_size=self.batch_size,
            epochs=self.epochs,
        )
        cached_model_predict_clear()

    def predict(self, data):
        """Predicts target values.

        Args:
            data: the samples to predict the target values for.

        Returns: the predicted traget values.

        """
        data = self._reshape_data(data)
        return cached_model_predict(self.model, data)

    @abstractmethod
    def _create_model(self, input_shape, output_shape):
        self.model = None

    @abstractmethod
    def _data_shapes(self, data, target):
        pass

    @abstractmethod
    def _reshape_data(self, data):
        pass
