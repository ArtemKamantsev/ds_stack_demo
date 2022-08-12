from typing import Any
from unittest import TestCase

import numpy as np


class TestInteroperability(TestCase):
    def test_array_interface(self) -> None:
        array: np.ndarray = np.arange(6).reshape((2, 3))
        # pylint: disable=no-member
        array_interface: dict[str, Any] = array.__array_interface__.copy()
        array_interface['shape'] = (3, 2)

        class CustomDataView:
            __array_interface__: dict[str, Any]

            def __init__(self, interface: dict[str, Any]):
                self.__array_interface__ = interface

        custom_view: CustomDataView = CustomDataView(array_interface)
        # noinspection PyTypeChecker, PyArgumentList
        array_from_view: np.ndarray = np.array(custom_view, copy=False)
        array_from_view[0, 0] = 42

        self.assertEqual(array_from_view.shape, (3, 2))
        # pylint: disable=unsubscriptable-object
        self.assertEqual(array[0, 0], 42)
