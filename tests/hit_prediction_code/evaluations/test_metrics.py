# -*- coding: utf-8 -*-
"""Modules containing evaluation metrics tests."""
import unittest

import numpy as np

from hit_prediction_code.evaluations.metrics import hit_nonhit_accuracy_score


class FakeEstimator():
    """Allows to hardcode predictions."""

    def __init__(self, predictions):
        """Intitializes the estimator with the predefined prediction.

        Args:
            predictions: used as predefined predictions when calling predict.

        """
        self.predictions = predictions

    def predict(self, x):
        """Returns the predefined predictions.

        Args:
            x: input to predict target values. This is ignored.

        """
        return self.predictions


class TestHitNonHitAccuracyScore(unittest.TestCase):
    """Tests for the hit_nonhit_accuracy_score."""

    def test_all_correct(self):
        """Test if the score works for all correct predictions."""
        predictions = np.array([
            1,
            2,
            3,
            50,
            100,
            101,
            15,
            150,
        ])

        estimator = FakeEstimator(predictions)

        expected = np.array([
            100,
            100,
            100,
            100,
            100,
            101,
            100,
            101,
        ])

        self.assertEqual(
            hit_nonhit_accuracy_score(
                estimator=estimator,
                x=None,
                y=expected,
            ),
            1.0,
        )

    def test_all_wrong(self):
        """Test if the score works for all wrong predictions."""
        predictions = np.array([
            1,
            2,
            3,
            50,
            100,
            101,
            15,
            150,
        ])

        estimator = FakeEstimator(predictions)

        expected = np.array([
            101,
            101,
            101,
            101,
            101,
            100,
            101,
            100,
        ])

        self.assertEqual(
            hit_nonhit_accuracy_score(
                estimator=estimator,
                x=None,
                y=expected,
            ),
            0.0,
        )

    def test_some_correct(self):
        """Test if the score works for 25% wrong predictions."""
        predictions = np.array([
            1,
            2,
            3,
            50,
            100,
            101,
            15,
            150,
        ])

        estimator = FakeEstimator(predictions)

        expected = np.array([
            100,
            101,
            100,
            101,
            100,
            101,
            100,
            101,
        ])

        self.assertEqual(
            hit_nonhit_accuracy_score(
                estimator=estimator,
                x=None,
                y=expected,
            ),
            0.75,
        )
