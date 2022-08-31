from unittest import TestCase
from typing import Any
import numpy as np


# pylint: disable=unsubscriptable-object
class TestMiscellaneous(TestCase):
    def test_nan(self) -> None:
        # pylint: disable=comparison-with-callable
        array: np.ndarray = np.array([np.nan, 42])

        self.assertEqual(len(array[array == np.nan]), 0)
        self.assertFalse(np.nan == np.nan)
        self.assertTrue(np.isnan(array[0]))
        self.assertEqual(np.nan_to_num(array[0]), 0)
        # pylint: disable=no-member
        self.assertTrue(np.isnan(array.sum()))
        self.assertEqual(np.nansum(array), 42)

    def test_inf(self) -> None:
        array: np.ndarray = np.array([-np.inf, 42, np.inf])

        self.assertTrue(np.inf == np.inf)
        self.assertTrue(np.isinf(array[0]))
        self.assertTrue(np.isfinite(array[1]))
        self.assertTrue(np.isinf(array[2]))

    def test_numerical_exceptions(self):
        # pylint: disable=unused-variable
        old_settings: dict[Any, Any] = np.seterr(all='raise')

        with self.assertRaises(FloatingPointError):
            result: np.array = np.zeros((1,)) / 0
