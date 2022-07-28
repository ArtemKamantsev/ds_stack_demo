from abc import ABC, abstractmethod
from numbers import Number
from unittest import TestCase


class TestClass(TestCase):
    def test_default_inheritance(self) -> None:
        class A:
            pass

        self.assertIsInstance(A(), object)

    def test_diamond_inheritance_methods_search(self) -> None:
        class DataSource(ABC):
            @abstractmethod
            def get_data(self) -> Number:
                raise NotImplementedError

        class Interactor(DataSource, ABC):
            def get_data_processed(self, factor: Number) -> Number:
                data = self.get_data()

                return data * factor

        class DataSourceMockData(DataSource):
            def get_data(self) -> Number:
                return 42

        class MultiplicatorMock(Interactor, DataSourceMockData):
            pass

        self.assertEqual(MultiplicatorMock().get_data_processed(1), 42)
