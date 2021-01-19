"""Test for the convert_to_closest_label converter."""
import pytest

from hit_prediction_code.transformers.label import convert_to_closest_label

EPSILON = 1 / 1_000_000


def test_to_short_label_list():
    """Tests if exception is raised for a to short label list."""
    with pytest.raises(ValueError):
        convert_to_closest_label(42, [1])


def test_value_smaller_than_first_class():
    """Tests if the smallest class is returned correctly."""
    assert convert_to_closest_label(-EPSILON, [0, 1, 2, 4]) == 0


def test_value_of_first_class():
    """Tests if a value between first two classes is assigned correctly."""
    assert convert_to_closest_label(.5 - EPSILON, [0, 1, 2, 4]) == 0


def test_value_of_second_class():
    """Tests if a value between first two classes is assigned correctly."""
    assert convert_to_closest_label(.5, [0, 1, 2, 4]) == 1


def test_value_of_third_class():
    """Tests if a value between last two classes is assigned correctly."""
    assert convert_to_closest_label(3 - EPSILON, [0, 1, 2, 4]) == 2


def test_value_of_fourth_class():
    """Tests if a value between last two classes is assigned correctly."""
    assert convert_to_closest_label(3, [0, 1, 2, 4]) == 4


def test_value_bigger_than_last_class():
    """Tests if the biggest class is returned correctly."""
    assert convert_to_closest_label(4 + EPSILON, [0, 1, 2, 4]) == 4
