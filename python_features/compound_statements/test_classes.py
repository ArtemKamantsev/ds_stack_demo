from abc import ABC, abstractmethod
from unittest import TestCase


class TestClass(TestCase):
    def test_default_inheritance(self):
        class A:
            pass

        self.assertIsInstance(A(), object)

    def test_diamond_inheritance_methods_search(self):
        class DataSource(ABC):
            @abstractmethod
            def get_data(self):
                raise NotImplementedError

        class Multiplicator(DataSource, ABC):
            def multiply_data(self, factor):
                data = self.get_data()

                return data * factor

        class DataSourceMockData(DataSource):
            def get_data(self):
                return 42

        class MultiplicatorMock(Multiplicator, DataSourceMockData):
            pass

        self.assertEqual(MultiplicatorMock().multiply_data(1), 42)
