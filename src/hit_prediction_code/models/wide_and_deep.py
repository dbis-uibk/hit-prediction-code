"""Module containing implementations of the wide and deep model."""
import numpy as np
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from ..transformers.label import convert_array_to_class_vector
from .building_blocks import HitPredictionModel
from .building_blocks import dense_layers
from .building_blocks import get_initializer


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
                 label_output=False,
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
            label_output: decides if the output is a vector of labels.
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
        self.label_output = label_output

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
        output = Dense(
            output_shape,
            activation=self.output_activation,
            kernel_initializer=get_initializer(self.output_activation),
            name='output',
            use_bias=use_bias,
        )(dense_layer)

        self.model = Model(inputs=input_list, outputs=output)

    def _data_shapes(self, data, labels):
        try:
            return None, labels.shape[1]
        except IndexError:
            return None, 1

    def _reshape_data(self, data):
        features = []
        for index, _ in self.features:
            feature = data[..., index]
            features.append(feature)
        return features


class WideAndDeepOrdinal(WideAndDeep):
    """WideAndDeep model trained on ordinal encoded classes."""

    def __init__(self,
                 labels,
                 loss='mse',
                 optimizer='adam',
                 metrics=None,
                 deep_activation='sigmoid',
                 dense_activation='relu',
                 output_activation='sigmoid',
                 epochs=1,
                 batch_size=None,
                 features=None,
                 batch_normalization=False,
                 dropout_rate=None,
                 dense_output_size=None,
                 num_dense_layer=2,
                 label_output=True,
                 predict_strategy='relative',
                 vectorization_strategy='fill',
                 **kwargs):
        """Initializes the model.

        Args:
            labels (list): labels used to encode the input labels.
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
            label_output: ignored; predictions are always single labels.
            predict_strategy: defines how the prediction gets selected.
                * 'relative': is the default and selects the class based on
                  the propability gain between two classes. For this strategy
                  the labels need to use a 'fill' strategy.
                * 'class_distribution' computes the difference between the
                  expected probability and the learned label distribution.
                * 'argmax' returns the class with the max probability.
            vectorization_strategy (str): the strategie used to convert
                regression values to class labels.
            kwargs: key-value arguments passed to the super constructor.
        """
        super().__init__(**kwargs)

        self.labels = labels
        if metrics is None:
            self.metrics = ['binary_crossentropy']
        else:
            self.metrics = metrics

        # the base model already supports that by setting label_output to True.
        if predict_strategy == 'argmax':
            self.label_output = True
        else:
            self.label_output = False

        self.input_list = []
        self.loss = loss
        self.optimizer = optimizer

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
        self.predict_strategy = predict_strategy
        self.vectorization_strategy = vectorization_strategy

        self._sample_count = 0
        self._class_count = 0

    @property
    def labels(self):
        """Property specifying the used labels."""
        return self._config.get('labels')

    @labels.setter
    def labels(self, value):
        self._config['labels'] = value

    @property
    def predict_strategy(self):
        """Property specifying the used prediction strategy."""
        return self._config['predict_strategy']

    @predict_strategy.setter
    def predict_strategy(self, value):
        if value in ['relative', 'class_distribution', 'argmax']:
            self._config['predict_strategy'] = value
        else:
            raise ValueError(f'Predict strategy \'{value}\' unknown.')

    @property
    def vectorization_strategy(self):
        """Property specifying the used vectorization strategy."""
        return self._config['vectorization_strategy']

    @vectorization_strategy.setter
    def vectorization_strategy(self, value):
        self._config['vectorization_strategy'] = value

    def fit(self, data, target, epochs=None):
        """Converts the target labels to class vectors before fitting."""
        target = convert_array_to_class_vector(
            target,
            self.labels,
            strategy=self.vectorization_strategy,
        )

        self._sample_count += len(target)
        self._class_count += np.sum(target, axis=0)

        super().fit(data, target, epochs=epochs)

    def predict(self, data):
        """Predicts an ordinal value."""
        prediction = super().predict(data)

        if self.predict_strategy == 'argmax':
            predicted = prediction
        elif self.predict_strategy == 'relative':
            predicted = self._get_relative_proba(prediction)
        elif self.predict_strategy == 'class_distribution':
            predicted = self._get_class_distribution_proba(prediction)
        else:
            raise ValueError(
                f'Strategy \'{self.predict_strategy}\' not implemented.')

        return np.argmax(predicted, axis=1)

    def _get_relative_proba(self, prediction):
        relative = []
        for i in range(len(self.labels)):
            if i < len(self.labels) - 1:  # all except the last
                relative.append(prediction[:, i] - prediction[:, i + 1])
            else:  # the last
                relative.append(prediction[:, i])

        return np.vstack(relative).T

    def _get_class_distribution_proba(self, prediction):
        class_proba = self._class_count / self._sample_count

        return prediction - class_proba
