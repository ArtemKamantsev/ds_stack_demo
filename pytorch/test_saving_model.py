from os.path import join
from unittest import TestCase

import torch
from torch import Tensor
from torch.nn import Parameter

from constants import OUTPUT_PATH

_save_path: str = join(OUTPUT_PATH, 'model.pth')


class CustomModel(torch.nn.Module):
    __param: Parameter

    def __init__(self):
        super().__init__()
        self.__param = Parameter(torch.empty(1, 1))
        torch.nn.init.xavier_uniform_(self.__param)

    def forward(self, x: Tensor) -> Tensor:
        return x * self.__param


class TestSavingModel(TestCase):
    # noinspection PyUnresolvedReferences
    def test_saving_model_state(self) -> None:
        data: Tensor = torch.tensor([1]).float()
        model = torch.nn.Sequential(
                torch.nn.Linear(1, 1)
        )
        prediction_first: Tensor = model(data)

        torch.save(model.state_dict(), _save_path)
        model_reloaded = torch.nn.Sequential(
                torch.nn.Linear(1, 1)
        )
        prediction_second: Tensor = model_reloaded(data)

        model_reloaded.load_state_dict(torch.load(_save_path))
        prediction_second_reloaded: Tensor = model_reloaded(data)

        self.assertFalse((prediction_first == prediction_second).all().item())
        self.assertTrue((prediction_first == prediction_second_reloaded).all().item())

    def test_saving_model(self) -> None:
        data: Tensor = torch.tensor([1]).float()

        model = torch.nn.Sequential(
                torch.nn.Linear(1, 1)
        )
        prediction: Tensor = model(data)
        torch.save(model, _save_path)

        model_reloaded: torch.nn.Sequential = torch.load(_save_path)
        prediction_reloaded: Tensor = model_reloaded(data)

        self.assertTrue((prediction == prediction_reloaded).all().item())

    def test_saving_custom_model(self) -> None:
        # pylint: disable=invalid-name, global-statement
        global CustomModel
        data: Tensor = torch.tensor([1])
        model = CustomModel()
        prediction: Tensor = model(data)

        torch.save(model, _save_path)
        model_reloaded: CustomModel = torch.load(_save_path)
        prediction_reloaded: Tensor = model_reloaded(data)

        del CustomModel
        with self.assertRaises(AttributeError):
            # pylint: disable=unused-variable
            model_reloaded2: CustomModel = torch.load(_save_path)

        self.assertTrue((prediction == prediction_reloaded).all().item())
