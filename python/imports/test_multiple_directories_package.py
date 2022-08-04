from unittest import TestCase

# pylint: disable=no-name-in-module, import-error
from python.imports.part1.variable import variable

__all__ = ['TestMultipleDirectories']


class TestMultipleDirectories(TestCase):
    def test_variable_imported(self) -> None:
        self.assertEqual(variable, 42)
