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

    def _test_accuracy_score(self,
                             predictions,
                             expected_predictions,
                             expected_score,
                             normalize=None,
                             threshold=None):
        predictions = np.array(predictions)
        expected_predictions = np.array(expected_predictions)
        estimator = FakeEstimator(predictions)

        if normalize is not None and threshold is not None:
            actual_score = hit_nonhit_accuracy_score(
                estimator=estimator,
                x=None,  # x is ignored by FakeEstimator
                y=expected_predictions,
                normalize=normalize,
                threshold=threshold,
            )
        elif normalize is not None:
            actual_score = hit_nonhit_accuracy_score(
                estimator=estimator,
                x=None,  # x is ignored by FakeEstimator
                y=expected_predictions,
                normalize=normalize,
            )
        elif threshold is not None:
            actual_score = hit_nonhit_accuracy_score(
                estimator=estimator,
                x=None,  # x is ignored by FakeEstimator
                y=expected_predictions,
                threshold=threshold,
            )
        else:
            actual_score = hit_nonhit_accuracy_score(
                estimator=estimator,
                x=None,  # x is ignored by FakeEstimator
                y=expected_predictions,
            )

        self.assertEqual(actual_score, expected_score)

    def test_all_correct(self):
        """Test if the score works for all correct predictions."""
        self._test_accuracy_score(
            predictions=[
                1,
                2,
                3,
                50,
                100,
                101,
                15,
                150,
            ],
            expected_predictions=[
                100,
                100,
                100,
                100,
                100,
                101,
                100,
                101,
            ],
            expected_score=1.0,
        )

    def test_all_wrong(self):
        """Test if the score works for all wrong predictions."""
        self._test_accuracy_score(
            predictions=[
                1,
                2,
                3,
                50,
                100,
                101,
                15,
                150,
            ],
            expected_predictions=[
                101,
                101,
                101,
                101,
                101,
                100,
                101,
                100,
            ],
            expected_score=0.0,
        )

    def test_some_correct(self):
        """Test if the score works for 25% wrong predictions."""
        self._test_accuracy_score(
            predictions=[
                1,
                2,
                3,
                50,
                100,
                101,
                15,
                150,
            ],
            expected_predictions=[
                100,
                101,
                100,
                101,
                100,
                101,
                100,
                101,
            ],
            expected_score=0.75,
        )

    def test_normalize(self):
        """Test if the normalize parameter is able to switch to counting."""
        self._test_accuracy_score(
            predictions=[
                1,
                2,
                3,
                50,
                100,
                101,
                15,
                150,
            ],
            expected_predictions=[
                100,
                100,
                100,
                100,
                100,
                101,
                100,
                101,
            ],
            expected_score=8,
            normalize=False,
        )

    def test_threshold(self):
        """Test if the threshold with a changed value works as expected."""
        self._test_accuracy_score(
            predictions=[
                1,
                2,
                3,
                50,
                100,
                145,
                140,
                141,
            ],
            expected_predictions=[
                100,
                100,
                100,
                100,
                100,
                150,
                141,
                140,
            ],
            expected_score=0.75,
            threshold=140,
        )

    def test_normalize_threshold(self):
        """Test if threshold works as expected without normalize."""
        self._test_accuracy_score(
            predictions=[
                1,
                2,
                3,
                50,
                100,
                145,
                140,
                141,
            ],
            expected_predictions=[
                100,
                100,
                100,
                100,
                100,
                150,
                141,
                140,
            ],
            expected_score=6,
            normalize=False,
            threshold=140,
        )
