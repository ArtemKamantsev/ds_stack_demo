from unittest import TestCase
from typing import Literal, TypeVar, Protocol, runtime_checkable


class TestTypeHints(TestCase):
    def test_literal_not_hashable(self) -> None:
        non_hashable_value = {1: 2}
        l1 = Literal[non_hashable_value]
        l2 = Literal[non_hashable_value]
        with self.assertRaises(TypeError):
            # pylint: disable=unused-variable
            is_equal: bool = l1 == l2

    def test_runtime_checkable(self) -> None:
        T = TypeVar('T', bound=str)

        class Proto1(Protocol[T]):
            def f(self, arg: T) -> T:
                return arg

        @runtime_checkable
        class Proto2(Protocol[T]):
            def f(self, arg: T) -> T:
                return arg

        class Implementation:
            def f(self, arg: int) -> int:
                return arg

        with self.assertRaises(TypeError):
            isinstance(Implementation(), Proto1)
        # checks only the presence of the required methods, not their type signatures
        self.assertTrue(isinstance(Implementation(), Proto2))
