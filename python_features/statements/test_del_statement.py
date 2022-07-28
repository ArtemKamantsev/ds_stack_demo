from unittest import TestCase

__all__ = ['DelStatement']

global_value: int = 0


class DelStatement(TestCase):
    def setUp(self) -> None:
        global global_value
        global_value = 0

    def test_undeclared_variable(self) -> None:
        # Why does NameError raised instead of UnboundLocalError?
        # Out of testing environment UnboundLocalError is raised in such case
        with self.assertRaises(NameError):
            value: Any = local_value

    def test_on_local_variable(self) -> None:
        local_value: int = 0
        del local_value
        with self.assertRaises(NameError):
            value: Any = local_value

    def test_on_global_value(self) -> None:
        global global_value
        del global_value
        with self.assertRaises(NameError):
            value: Any = global_value

    def test_on_nested_free_variable_value(self) -> None:
        free_variable: int = 0

        def use_free_variable() -> None:
            value: int = free_variable

        use_free_variable()
        del free_variable
        with self.assertRaises(NameError, msg='Changed in version 3.2: Previously it was illegal to delete a name from '
                                              'the local namespace if it occurs as a free variable in a nested block.'):
            value: int = free_variable
