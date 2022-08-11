from unittest import TestCase

import numpy as np
from scipy.ndimage import correlate1d, convolve1d


class TestFilteringFunctions(TestCase):
    def test_correlate1d(self) -> None:
        data: list[int] = [0, 0, 1, 1, 1, 0, 0]

        self.assertTrue(np.all(correlate1d(data, np.array([-1, 1])) == np.array([0, 0, 1, 0, 0, -1, 0])))
        # shift result left by 1
        self.assertTrue(np.all(correlate1d(data, np.array([-1, 1]), origin=-1) == np.array([0, 1, 0, 0, -1, 0, 0])))
        # the same effect as in line above, but less computationally effective
        self.assertTrue(np.all(correlate1d(data, np.array([0, -1, 1])) == np.array([0, 1, 0, 0, -1, 0, 0])))

    def test_convolve1d(self) -> None:
        data: list[int] = [0, 0, 1, 1, 1, 0, 0]

        # convolve1d is the same as correlate1d but performed with the rotated kernel
        # that is why origin behaviour is opposite
        self.assertTrue(np.all(convolve1d(data, np.array([1, -1])) == np.array([0, 1, 0, 0, -1, 0, 0])))
        # shift result right by 1
        self.assertTrue(np.all(convolve1d(data, np.array([1, -1]), origin=-1) == np.array([0, 0, 1, 0, 0, -1, 0])))
        # the same effect as in line above, but less computationally effective
        self.assertTrue(np.all(convolve1d(data, np.array([0, 1, -1])) == np.array([0, 0, 1, 0, 0, -1, 0])))
