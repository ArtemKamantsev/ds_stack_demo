from unittest import TestCase

import numpy as np


class TestNdArrays(TestCase):
    def test_matrix_product(self) -> None:
        matrix1: np.ndarray = np.empty(2, dtype=int)
        matrix2: np.ndarray = np.empty(2, dtype=int)
        matrix3: np.ndarray = np.empty(2, dtype=np.int64)

        product1: np.ndarray = matrix1.dot(matrix2)
        product2: np.ndarray = matrix1 @ matrix2
        product3: np.ndarray = matrix1 @ matrix3

        self.assertEqual(product1, product2)
        self.assertEqual(product1.dtype, product2.dtype)
        self.assertEqual(product2.dtype, np.int32, msg='Despite args had python\'s int type, it changed to np.int32')
        self.assertEqual(product3.dtype, np.int64, msg='Type broadcasts to most general type of 2 types given')

    def test_flat(self) -> None:
        array_flat: np.ndarray = np.zeros(4, dtype=int)
        array_2d: np.ndarray = np.zeros((2, 2), dtype=int)

        self.assertTrue(np.all(array_flat == array_2d.flat))

    def test_boolean_indexing(self) -> None:
        array: np.ndarray = np.arange(4).reshape(2, 2)
        mask: np.ndarray = np.array([[True, False],
                                     [False, True]])
        result: np.ndarray = array[mask]

        self.assertIsNone(result.base)  # advanced indexing always returns a copy
        self.assertEqual(result.ndim, 1)
        self.assertEqual(list(result), [0, 3])

    def test_integer_indexing(self) -> None:
        array: np.ndarray = np.arange(4).reshape(2, 2)
        index_dim1: np.ndarray
        index_dim2: np.ndarray
        # pylint: disable=unbalanced-tuple-unpacking
        index_dim1, index_dim2 = np.ix_([1, 1], [0, 0])
        result: np.ndarray = array[index_dim1, index_dim2]

        self.assertIsNone(result.base)  # advanced indexing always returns a copy
        self.assertEqual(result.ndim, 2)
        self.assertEqual(result.shape, (2, 2))
        self.assertTrue(np.all(result.flat == 2))

    def test_combined_indexing_first_case(self) -> None:
        array: np.ndarray = np.arange(4).reshape(2, 2)
        result: np.ndarray = array[[1, 1], 0:1]

        self.assertIsNone(result.base)  # advanced indexing always returns a copy
        self.assertEqual(result.ndim, 2)
        self.assertEqual(result.shape, (2, 1))
        self.assertTrue(np.all(result.flat == 2))

    def test_combined_indexing_second_case(self) -> None:
        array: np.ndarray = np.arange(12).reshape((2, 2, 3))
        result: np.ndarray = array[..., [[0]], :]

        self.assertIsNotNone(result.base)  # the view returned because basic indexing applied first
        self.assertEqual(result.ndim, 4)
        self.assertEqual(result.shape, (2, 1, 1, 3))
        self.assertTrue(np.all(result.flat == np.array([0, 1, 2, 6, 7, 8])))

    def test_indexing_broadcasting(self) -> None:
        array: np.ndarray = np.arange(4).reshape(2, 2)
        result: np.ndarray = array[[[1]], [0]]

        self.assertIsNone(result.base)  # advanced indexing always returns a copy
        self.assertEqual(result.ndim, 2)
        self.assertEqual(result.shape, (1, 1))
        self.assertEqual(result[0][0], 2)

    def test_assignment(self) -> None:
        array: np.ndarray = np.arange(2)
        array[np.zeros(2, dtype=int)] += 1

        self.assertEqual(array[0], 1)

    def test_manual_reshape(self) -> None:
        data: np.ndarray = np.arange(10)
        data.shape = (2, 5)

        self.assertEqual(data[1, 0], 5)

    def test_new_axis(self) -> None:
        data: np.ndarray = np.arange(4)

        data_view_1: np.ndarray = data[:, np.newaxis]
        data_view_2: np.ndarray = data[:, None]

        self.assertIsNone(np.newaxis)
        self.assertEqual(data_view_1.ndim, 2)
        self.assertEqual(data_view_1.shape, (4, 1))
        self.assertEqual(data_view_1.shape, data_view_2.shape)

    def test_broadcasting(self) -> None:
        array_first: np.ndarray = np.empty((5, 1))
        array_second: np.ndarray = np.empty((5, 1, 5))
        result: np.ndarray = array_first + array_second

        self.assertEqual(result.shape, (5, 5, 5))

    def test_broadcasting_error(self) -> None:
        array_first: np.ndarray = np.empty((5, 2))
        array_second: np.ndarray = np.empty((5, 1, 5))

        with self.assertRaises(ValueError):
            # pylint: disable=unused-variable
            result: np.ndarray = array_first + array_second

    def test_masked_array(self):
        data: list[int | None] = [1, np.nan, 3]
        regular_array = np.array(data)
        masked_array: np.ma.masked_array = np.ma.masked_array(data, mask=[False, True, False])

        self.assertTrue(np.isnan(regular_array.sum()))
        self.assertEqual(masked_array.sum(), 4)
