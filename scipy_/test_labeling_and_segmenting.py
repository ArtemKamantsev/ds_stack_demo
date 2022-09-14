from unittest import TestCase

import numpy as np
from scipy.ndimage import label, find_objects, sum_labels


class TestLabelingAndSegmenting(TestCase):
    image: np.ndarray

    def setUp(self) -> None:
        self.image = np.array([[0.5, 0],
                               [0, 0.5]])

    def test_labeling_4_connected(self):
        footprint_4_connected: list[list[int]] = [[0, 1, 0],
                                                  [1, 1, 1],
                                                  [0, 1, 0]]

        image_labeled: np.ndarray
        labels_count: int
        image_labeled, labels_count = label(self.image, footprint_4_connected)

        self.assertEqual(labels_count, 2)
        self.assertTrue(np.all(image_labeled == np.array([[1, 0],
                                                          [0, 2]])))

    def test_labeling_8_connected(self):
        footprint_8_connected: list[list[int]] = [[1, 1, 1],
                                                  [1, 1, 1],
                                                  [1, 1, 1]]

        image_labeled: np.ndarray
        labels_count: int
        image_labeled, labels_count = label(self.image, footprint_8_connected)

        self.assertEqual(labels_count, 1)
        self.assertTrue(np.all(image_labeled == np.array([[1, 0],
                                                          [0, 1]])))

    def test_objects_detection(self):
        image_labeled: np.ndarray
        # pylint: disable=unused-variable
        labels_count: int
        image_labeled, labels_count = label(self.image)

        # noinspection PyTypeChecker
        objects_list: list[slice] = find_objects(image_labeled)

        self.assertEqual(objects_list[0], (slice(0, 1, None), slice(0, 1, None)))
        self.assertEqual(objects_list[1], (slice(1, 2, None), slice(1, 2, None)))

        obj_1: np.ndarray = self.image[objects_list[0]]
        obj_2: np.ndarray = self.image[objects_list[1]]
        self.assertEqual(obj_1.shape, (1, 1))
        self.assertEqual(obj_2.shape, (1, 1))

    def test_object_sizes_measurement(self):
        image_labeled: np.ndarray
        # pylint: disable=unused-variable
        labels_count: int
        image_labeled, labels_count = label(self.image)
        # image_labeled legend:
        # 0 - background
        # 1 - first_obj
        # 2 - second_obj

        # noinspection PyTypeChecker
        objects_list: list[slice] = find_objects(image_labeled)

        square: np.ndarray = np.ones(self.image.shape)
        first_object_area: float = sum_labels(square[objects_list[0]], image_labeled[objects_list[0]], 1)
        self.assertEqual(first_object_area, 1.0)

        all_object_areas: np.ndarray = sum_labels(square, image_labeled, [0, 1, 2])
        self.assertTrue(np.all(all_object_areas == np.array([2.0, 1.0, 1.0])))
