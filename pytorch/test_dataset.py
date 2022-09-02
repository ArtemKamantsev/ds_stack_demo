from typing import Callable
from unittest import TestCase

import torch
from torch.utils.data import Dataset, DataLoader


class TestDataset(TestCase):
    def test_custom_dataset(self) -> None:
        class RandomDataset(Dataset):
            __data: torch.Tensor
            __labels: torch.Tensor
            __transform: Callable[[torch.Tensor], torch.Tensor]
            __target_transform: Callable[[torch.Tensor], torch.Tensor]

            def __init__(self, data_shape: tuple[int, ...], seed: int = 42, transform=None, target_transform=None):
                torch.manual_seed(seed)
                self.__data = torch.randint(0, 2, data_shape)
                self.__labels = torch.randint(0, 2, (data_shape[0],))
                self.__transform = transform
                self.__target_transform = target_transform

            def __len__(self):
                return len(self.__labels)

            def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
                item: torch.Tensor = self.__data[idx]
                label: torch.Tensor = self.__labels[idx]
                if self.__transform:
                    item = self.__transform(item)
                if self.__target_transform:
                    label = self.__target_transform(label)

                return item, label

        random_dataset = RandomDataset((4, 1))
        data_loader = DataLoader(random_dataset, batch_size=2, shuffle=False)

        batch_list: list[list[torch.Tensor]] = list(iter(data_loader))
        self.assertEqual(len(batch_list), 2)

        batch1: list[torch.Tensor]
        batch2: list[torch.Tensor]
        data1: torch.Tensor
        label1: torch.Tensor
        data2: torch.Tensor
        label2: torch.Tensor

        batch1, batch2 = batch_list
        data1, label1 = batch1
        data2, label2 = batch2

        self.assertTrue((data1.numpy() == [[0], [1]]).all().all())
        self.assertTrue((label1.numpy() == [0, 1]).all().all())

        self.assertTrue((data2.numpy() == [[0], [0]]).all().all())
        self.assertTrue((label2.numpy() == [0, 0]).all().all())
