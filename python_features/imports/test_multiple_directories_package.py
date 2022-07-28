from unittest import TestCase

from python_features.imports.part1.variable import variable

__all__ = ['TestMultipleDirectories']


class TestMultipleDirectories(TestCase):
    def test_variable_imported(self) -> None:
        self.assertEqual(variable, 42)
