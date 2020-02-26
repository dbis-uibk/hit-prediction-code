# -*- coding: utf-8 -*-
"""Module containing implementations of the wide and deep model."""
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from .building_blocks import HitPredictionModel
from .building_blocks import dense_layers


class WideAndDeep(HitPredictionModel):
    """Wide and Deep Model designed for hit song prediction."""

    def __init__(self,
                 loss='mse',
                 optimizer='adam',
                 metrics=None,
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
        """Initializes the Wide and Deep Model object.

        Args:
            loss: the loss function used to train the network.
            optimizer: the optimizer used to train the model.
            metrics: a list of metrics used to evaluate the model during
                training. If set to None, MAE is used.
            deep_activation: the activation function used for the deep part of
                the network.
            dense_activation: the activation function used for the dense part.
            output_activation: the activation function used for the output.
            epochs: the number of epochs used during training.
            batch_size: the batch size used to train the model.
            features: a list of tuples describing the features used for
                training and the network part (wide or deep) that is used to
                handle them.
            batch_normalization: configures if batch normalization is used for
                the dense network part.
            dropout_rate: the dropout rate used for the dense part.
            dense_output_size: the output width of the dense layers.
            num_dense_layer: the number of dense layers in the dense part.
            kwargs: key-value arguments passed to the super constructor.

        """
        super().__init__(**kwargs)

        self.input_list = []
        self.loss = loss
        self.optimizer = optimizer

        if metrics is None:
            self.metrics = ['mae']
        else:
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

    @property
    def deep_activation(self):
        """Property specifying the deep activation function."""
        return self._config.get('deep_activation')

    @deep_activation.setter
    def deep_activation(self, value):
        self._config['deep_activation'] = value

    @property
    def dense_output_size(self):
        """Property specifying the output size of the dense layers."""
        return self._config.get('dense_output_size')

    @dense_output_size.setter
    def dense_output_size(self, value):
        self._config['dense_output_size'] = value

    @property
    def features(self):
        """Property specifying the features that are used."""
        return self._config.get('features')

    @features.setter
    def features(self, value):
        self._config['features'] = value

    def _create_model(self, input_shape, output_shape):
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

        dense_layer = dense_layers(
            self.batch_normalization,
            self.dropout_rate,
            dense_output_size,
            self.num_dense_layer,
            self.dense_activation,
            concat_tensor,
        )

        use_bias = not self.batch_normalization
        output = Dense(output_shape,
                       activation=self.output_activation,
                       name='output',
                       use_bias=use_bias)(dense_layer)

        self.model = Model(inputs=input_list, outputs=output)

    def _data_shapes(self, data, labels):
        return None, 1

    def _reshape_data(self, data):
        features = []
        for index, _ in self.features:
            feature = data[..., index]
            features.append(feature)
        return features
