from abc import ABC, abstractmethod
from numbers import Number
from unittest import TestCase


class TestClass(TestCase):
    def test_class_variable_scope(self) -> None:
        class A:
            class_variable = 0

            def f(self_inner) -> None:
                # The scope of names defined in a class block is limited to the class block;
                # it does not extend to the code blocks of methods
                with self.assertRaises(NameError):
                    variable = class_variable

        A().f()

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
