"""Module containing implementations of the simple neural network model."""
from tensorflow.keras.layers import AlphaDropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from .building_blocks import HitPredictionModel
from .building_blocks import get_initializer


class SimpleNN(HitPredictionModel):
    """Simple neural network model for hit song prediction."""

    def __init__(self,
                 loss='mse',
                 optimizer='adam',
                 metrics=None,
                 dense_activation='selu',
                 output_activation=None,
                 dense_sizes=(100,),
                 epochs=1,
                 batch_size=100,
                 features=None,
                 dropout_rate=0.1,
                 **kwargs):
        """Initializes the Wide and Deep Model object.

        Args:
            loss: the loss function used to train the network.
            optimizer: the optimizer used to train the model.
            metrics: a list of metrics used to evaluate the model during
                training. If set to None, MAE is used.
            dense_activation: the activation function used for the dense part.
            output_activation: the activation function used for the output.
            dense_sizes: the dense layer sizes.
            epochs: the number of epochs used during training.
            batch_size: the batch size used to train the model.
            features: a list of tuples describing the features used for
                training and the network part (wide or deep) that is used to
                handle them.
            dropout_rate: the dropout rate used for the dense part.
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
        self.dense_activation = dense_activation
        self.output_activation = output_activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.features = features
        self.dropout_rate = dropout_rate
        self.num_dense_layer = len(dense_sizes)
        self.dense_sizes = dense_sizes

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

    @property
    def dense_sizes(self):
        """Property specifying the dense layer sizers."""
        return self._config.get('dense_sizes')

    @dense_sizes.setter
    def dense_sizes(self, value):
        self._config['dense_sizes'] = value

    def _create_model(self, input_shape, output_shape):
        # Input layer.
        hidden = Input(shape=input_shape)

        for size in self.dense_sizes:
            hidden = Dense(
                size,
                activation=self.dense_activation,
                kernel_initializer=get_initializer(self.output_activation),
            )(hidden)

            if self.dropout_rate is not None:
                hidden = AlphaDropout(self.dropout_rate)(hidden)

        output = Dense(output_shape, activation=self.output_activation)(hidden)
        self.model = Model(inputs=hidden, outputs=output)

    def _data_shapes(self, data, labels):
        return None, 1
