"""Tests for the WideAndDeepOrdinal class."""
import numpy as np
import pytest
from sklearn.base import clone

import hit_prediction_code.models.wide_and_deep
from hit_prediction_code.models.wide_and_deep import WideAndDeepOrdinal
from hit_prediction_code.transformers.label import \
    convert_array_to_class_vector


@pytest.mark.parametrize('cloning', [True, False])
def test_default_init_arg_passing(cloning):
    """Tests if the arguments are passed correctly to the super class."""
    test_labels = list(range(8))
    model = WideAndDeepOrdinal(labels=test_labels)

    if cloning:
        model = clone(model)

    assert model.labels == test_labels
    assert model.loss == 'mse'
    assert model.optimizer == 'adam'
    assert model.metrics == ['binary_crossentropy']
    assert model.deep_activation == 'sigmoid'
    assert model.dense_activation == 'relu'
    assert model.output_activation == 'sigmoid'
    assert model.epochs == 1
    assert model.batch_size is None
    assert model.features is None
    assert model.batch_normalization is False
    assert model.dropout_rate is None
    assert model.dense_output_size is None
    assert model.num_dense_layer == 2
    assert model.label_output is False
    assert model.predict_strategy == 'relative'


@pytest.mark.parametrize('cloning', [True, False])
def test_init_args_passing(cloning):
    """Tests if the arguments are passed correctly to the super class."""
    test_labels = list(range(8))
    model = WideAndDeepOrdinal(epochs=100,
                               labels=test_labels,
                               label_output=False,
                               predict_strategy='class_distribution')

    if cloning:
        model = clone(model)

    assert model.labels == test_labels
    assert model.loss == 'mse'
    assert model.optimizer == 'adam'
    assert model.metrics == ['binary_crossentropy']
    assert model.deep_activation == 'sigmoid'
    assert model.dense_activation == 'relu'
    assert model.output_activation == 'sigmoid'
    assert model.epochs == 100
    assert model.batch_size is None
    assert model.features is None
    assert model.batch_normalization is False
    assert model.dropout_rate is None
    assert model.dense_output_size is None
    assert model.num_dense_layer == 2
    assert model.label_output is False
    assert model.predict_strategy == 'class_distribution'


def test_init_predict_strategy_checks():
    """Tests if there are checks for unknown predict_strategy types."""
    with pytest.raises(ValueError, match='\'void\' unknown'):
        test_labels = list(range(8))
        WideAndDeepOrdinal(labels=test_labels, predict_strategy='void')


@pytest.mark.parametrize('epochs', [None, 1, 2])
def test_fitting(mocker, epochs):
    """Tests if the model is fit correctly."""
    mock_call = {'call_count': 0}

    def _mock_fit(self, data, target, epochs):
        mock_call['call_count'] += 1
        mock_call['data'] = data
        mock_call['target'] = target
        mock_call['epochs'] = epochs

    mocker.patch.object(
        hit_prediction_code.models.wide_and_deep.WideAndDeep,
        'fit',
        _mock_fit,
    )

    test_labels = list(range(8))
    test_data = [1, 2]
    test_target = np.array(2 * test_labels)
    test_label_counts = np.arange(8, 0, -1) * 2

    model = WideAndDeepOrdinal(labels=test_labels)
    model.fit(data=test_data, target=test_target, epochs=epochs)

    expected_target_labels = convert_array_to_class_vector(
        test_target,
        test_labels,
        strategy='fill',
    )

    assert mock_call['call_count'] == 1
    assert mock_call['data'] == test_data
    assert (mock_call['target'] == expected_target_labels).all()
    assert mock_call['epochs'] == epochs

    assert model._sample_count == len(test_target)
    assert (model._class_count == test_label_counts).all()


def test_predit_relative(mocker):
    """Tests if the model predicts correctly."""
    mock_call = {'call_count': 0}

    def _mock_predict(self, data):
        mock_call['call_count'] += 1
        mock_call['data'] = data

        return np.array([
            [1.0, 0.7, 0.6, 0.5],
            [1.0, 0.8, 0.4, 0.3],
            [1.0, 0.8, 0.7, 0.3],
            [1.0, 0.4, 0.3, 0.2],
        ])

    def _mock_fit(self, data, target, epochs):
        pass

    mocker.patch.object(
        hit_prediction_code.models.wide_and_deep.WideAndDeep,
        'fit',
        _mock_fit,
    )

    mocker.patch.object(
        hit_prediction_code.models.wide_and_deep.WideAndDeep,
        'predict',
        _mock_predict,
    )

    test_labels = list(range(4))
    test_data = [1, 2, 3, 4]
    expected_prediction = [3, 1, 2, 0]

    model = WideAndDeepOrdinal(
        labels=test_labels,
        predict_strategy='relative',
    )
    actual_prediction = model.predict(data=test_data)

    assert mock_call['call_count'] == 1
    assert mock_call['data'] == test_data
    assert (actual_prediction == expected_prediction).all()


@pytest.mark.parametrize('fit_calls', [1, 2, 4])
def test_predit_class_distribution(mocker, fit_calls):
    """Tests if the model predicts correctly."""
    mock_call = {'call_count': 0}

    def _mock_predict(self, data):
        mock_call['call_count'] += 1
        mock_call['data'] = data

        return np.array([
            [1.00, 0.70, 0.60, 0.50],
            [1.00, 0.85, 0.40, 0.30],
            [1.00, 0.80, 0.70, 0.30],
            [1.00, 0.40, 0.30, 0.20],
        ])

    def _mock_fit(self, data, target, epochs):
        pass

    mocker.patch.object(
        hit_prediction_code.models.wide_and_deep.WideAndDeep,
        'fit',
        _mock_fit,
    )

    mocker.patch.object(
        hit_prediction_code.models.wide_and_deep.WideAndDeep,
        'predict',
        _mock_predict,
    )

    test_labels = list(range(4))
    test_data = [1, 2]
    test_target = np.array(2 * test_labels)
    expected_prediction = [3, 1, 2, 0]

    model = WideAndDeepOrdinal(
        labels=test_labels,
        predict_strategy='class_distribution',
    )

    for _ in range(fit_calls):
        model.fit(data=test_data, target=test_target)

    actual_prediction = model.predict(data=test_data)

    assert mock_call['call_count'] == 1
    assert mock_call['data'] == test_data
    assert (actual_prediction == expected_prediction).all()

    class_proba = [1.0, 0.75, 0.5, 0.25]
    assert (model._class_count / model._sample_count == class_proba).all()


def test_predit_unknown(mocker):
    """Tests if the model predict strategy is checked correctly."""

    def _mock_predict(self, data):
        return np.array([
            [1.0, 0.7, 0.6, 0.5],
            [1.0, 0.8, 0.4, 0.3],
            [1.0, 0.8, 0.7, 0.3],
            [1.0, 0.4, 0.3, 0.2],
        ])

    mocker.patch.object(
        hit_prediction_code.models.wide_and_deep.WideAndDeep,
        'predict',
        _mock_predict,
    )

    test_labels = list(range(4))
    test_data = [1, 2, 3, 4]

    model = WideAndDeepOrdinal(labels=test_labels)

    # change internal state to circumvent setter checks
    model._config['predict_strategy'] = 'void'

    with pytest.raises(ValueError, match='\'void\' not implemented'):
        model.predict(data=test_data)
