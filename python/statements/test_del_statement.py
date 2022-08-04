from unittest import TestCase
from typing import Any

__all__ = ['DelStatement']

GLOBAL_VALUE: int = 0


class DelStatement(TestCase):
    def setUp(self) -> None:
        # pylint: disable=global-statement
        global GLOBAL_VALUE
        GLOBAL_VALUE = 0

    def test_undeclared_variable(self) -> None:
        # Why does NameError raised instead of UnboundLocalError?
        # Out of testing environment UnboundLocalError is raised in such case
        with self.assertRaises(NameError):
            # pylint: disable=undefined-variable, unused-variable
            value: Any = local_value

    def test_on_local_variable(self) -> None:
        local_value: int = 0
        del local_value
        with self.assertRaises(NameError):
            # pylint: disable=unused-variable
            value: Any = local_value

    def test_on_global_value(self) -> None:
        # pylint: disable=global-statement
        global GLOBAL_VALUE
        del GLOBAL_VALUE
        with self.assertRaises(NameError):
            # pylint: disable=unused-variable
            value: Any = GLOBAL_VALUE

    def test_on_nested_free_variable_value(self) -> None:
        free_variable: int = 0

        def use_free_variable() -> None:
            # pylint: disable=unused-variable
            value: int = free_variable

        use_free_variable()
        del free_variable
        with self.assertRaises(NameError, msg='Changed in version 3.2: Previously it was illegal to delete a name from '
                                              'the local namespace if it occurs as a free variable in a nested block.'):
            # pylint: disable=unused-variable
            value: int = free_variable
