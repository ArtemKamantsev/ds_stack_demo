from unittest import TestCase

import numpy as np


class TestStructuredDataTypeCreation(TestCase):
    data: list[tuple[int, int]]

    def setUp(self) -> None:
        self.data = [(0, 0), (42, 42)]

    def test_from_tuple(self) -> None:
        type1: np.dtype = np.dtype([(('abscissa', 'x'), int), (('ordinate', 'y'), int)])
        points: np.ndarray = np.array(self.data, dtype=type1)
        self.assertEqual(points[1]['x'], 42)
        self.assertEqual(points[1]['ordinate'], 42)

    def test_from_string(self) -> None:
        type2: np.dtype = np.dtype('int32, int32')
        points: np.ndarray = np.array(self.data, dtype=type2)
        self.assertEqual(points[1]['f1'], 42)

        type2.names = ('x', 'y')  # rename fields
        self.assertEqual(points[1]['x'], 42)

    def test_from_dict_case1(self) -> None:
        type3: np.dtype = np.dtype({
            'names': ['x', 'y'],
            'formats': [int, int],
        })
        points: np.ndarray = np.array(self.data, dtype=type3)
        self.assertEqual(points[1]['x'], 42)

    def test_from_dict_case2(self) -> None:
        type4: np.dtype = np.dtype({
            'x': (int, 0, 'abscissa'),
            'y': (int, 4, 'ordinate'),
        })
        points: np.ndarray = np.array(self.data, dtype=type4)
        self.assertEqual(points[1]['x'], 42)
        self.assertEqual(points[1]['ordinate'], 42)


class TestStructuredDataTypeAssignment(TestCase):
    points2d: np.ndarray

    def setUp(self) -> None:
        point1d_type: np.dtype = np.dtype([('x', int)])
        point2d_type: np.dtype = np.dtype([('x', int), ('y', int)])

        self.points2d: np.ndarray = np.array([(0, 0), (1, 1)], dtype=point2d_type)
        self.points1d: np.ndarray = np.array([(0,), (1,)], dtype=point1d_type)

    def test_tuple(self) -> None:
        self.points2d[0] = (42, 42)
        self.assertEqual(tuple(self.points2d[0]), (42, 42))

    def test_list(self) -> None:
        with self.assertRaises(ValueError):
            self.points2d[0] = [0, 0]

    def test_scalar(self) -> None:
        self.points2d[:] = 42
        self.assertEqual(tuple(self.points2d[0]), (42, 42))
        self.assertEqual(tuple(self.points2d[1]), (42, 42))

    def test_unstructured_to_structured(self) -> None:
        self.points2d[:] = [2, 42]
        self.assertEqual(tuple(self.points2d[0]), (2, 2))
        self.assertEqual(tuple(self.points2d[1]), (42, 42))

    def test_structured_to_unstructured(self) -> None:
        unstructured: np.ndarray = np.empty((2, 2), dtype=int)

        unstructured[:] = self.points1d  # broadcasting applied
        self.assertTrue(np.all(unstructured.flat == np.array([0, 1, 0, 1])))

        # assignment to unstructured array allowed only when structured array has single field
        with self.assertRaises(TypeError):
            unstructured[:] = self.points2d
