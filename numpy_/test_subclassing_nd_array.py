from numbers import Number
from typing import Any
from collections.abc import Sequence
from unittest import TestCase

import numpy as np


class FixedValue1DArray(np.ndarray):
    def __new__(cls, input_array: np.ndarray, value: Number) -> 'FixedValue1DArray':
        if input_array.ndim > 1:
            return NotImplemented

        obj: FixedValue1DArray = np.asarray(input_array).view(cls)
        obj._FixedValue1DArray__value = value
        obj.ensure_value()

        return obj

    # pylint: disable=unused-argument
    def __array_finalize__(self, obj: np.ndarray, *args: Sequence[Any], **kwargs: dict[Any, Any]) -> None:
        if obj is not None:
            # pylint: disable=attribute-defined-outside-init
            self.__value = getattr(obj, '_FixedValue1DArray__value', 42)

        self.ensure_value()

    def ensure_value(self) -> None:
        for i in range(self.shape[0]):
            self[i] = self.__value


class TestSubclassing(TestCase):
    data: np.array

    def setUp(self) -> None:
        self.data: np.array = np.arange(10)

    def test_subclassing_view(self) -> None:
        self.assertTrue(np.all(self.data.flat != 42))
        fixed_value_data: FixedValue1DArray = self.data.view(FixedValue1DArray)
        self.assertTrue(np.all(self.data.flat == 42))
        self.assertTrue(np.all(fixed_value_data.flat == 42))

    def test_subclassing_constructor(self) -> None:
        self.assertTrue(np.all(self.data.flat != 21))
        fixed_value_data: FixedValue1DArray = FixedValue1DArray(self.data, 21)
        self.assertTrue(np.all(self.data.flat == 21))
        self.assertTrue(np.all(fixed_value_data.flat == 21))

    def test_subclassing_slice(self) -> None:
        self.assertTrue(np.all(self.data.flat != 21))
        fixed_value_data: FixedValue1DArray = FixedValue1DArray(self.data, 21)[:5]
        self.assertTrue(np.all(self.data.flat == 21))
        self.assertTrue(np.all(fixed_value_data.flat == 21))
