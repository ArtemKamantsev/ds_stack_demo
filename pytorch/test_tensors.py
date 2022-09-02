from unittest import TestCase

import numpy as np
import torch


class TestTensors(TestCase):
    def test_standard_tensor_from_existing(self) -> None:
        data: list[list[int]] = [[1, 2],
                                 [3, 4]]
        tensor: torch.Tensor = torch.tensor(data)
        tensor_ones: torch.Tensor = torch.ones_like(tensor)

        self.assertTrue((tensor_ones.numpy() == [[1, 1],
                                                 [1, 1]]).all().all())

    def test_get_single_element_from_tensor(self) -> None:
        tensor: torch.Tensor = torch.tensor([[1, 2],
                                             [3, 4]])
        single_item_tensor: torch.Tensor = tensor.sum()

        self.assertEqual(len(single_item_tensor.shape), 0)
        self.assertEqual(single_item_tensor.item(), 10)
        self.assertIs(type(single_item_tensor.item()), int)

    def test_tensor_to_numpy(self) -> None:
        tensor_ones: torch.Tensor = torch.ones(3)
        numpy1: np.ndarray = tensor_ones.numpy()  # view
        numpy2: np.ndarray = np.asarray(tensor_ones)  # view
        numpy3: np.ndarray = np.array(tensor_ones)  # copy

        numpy1 += 1
        self.assertTrue((numpy1 == 2).all())
        self.assertTrue((numpy1 == tensor_ones.numpy()).all())
        self.assertTrue((numpy2 == 2).all())
        self.assertTrue((numpy3 == 1).all())

    def test_numpy_to_tensor(self) -> None:
        numpy: np.ndarray = np.ones(3)
        tensor: torch.Tensor = torch.from_numpy(numpy)

        numpy += 1
        self.assertTrue((numpy == 2).all())
        self.assertTrue((numpy == tensor.numpy()).all())
