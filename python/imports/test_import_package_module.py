from typing import Callable
from unittest import TestCase, TestSuite

from unittest.loader import TestLoader
# pylint: disable=wildcard-import, unused-wildcard-import
from python.statements.test_del_statement import *

temp: tuple[type[DelStatement]] = (DelStatement,)  # just to keep import during imports optimization operation


class Import(TestCase):
    def test_redundant_imports(self) -> None:
        with self.assertRaises(NameError):
            # pylint: disable=unused-variable
            value = GLOBAL_VALUE  # import avoided by using '__all__' attribute + import * syntax

    def test_name_binding(self) -> None:
        # pylint: disable=import-outside-toplevel
        from python.imports import sub1
        self.assertNotIn('sub2', dir(sub1))

        # pylint: disable=unused-import
        import python.imports.sub1.sub2
        self.assertIn('sub2', dir(sub1))

    def test_path(self) -> None:
        # pylint: disable=import-outside-toplevel
        import python as package
        import python.statements.test_del_statement as module

        self.assertIsNotNone(package.__path__, 'Packages - are modules and has a __path__ attribute')
        # Non package modules do not have a __path__ attribute
        with self.assertRaises(AttributeError):
            # pylint: disable=unused-variable, no-member
            path = module.__path__

    def test_name_package(self) -> None:
        # pylint: disable=import-outside-toplevel
        import python.statements
        import python.statements.test_del_statement as module

        self.assertEqual(python.__name__, python.__package__,
                         'The name of the top level package '
                         'that contains the module is bound '
                         'in the local namespace as a reference '
                         'to the top level package.')
        self.assertNotEqual(module.__name__, module.__package__)
        self.assertEqual(module.__package__, python.statements.__name__, 'Fully qualified name'
                                                                                  'must be used because no alias'
                                                                                  'was used')

    def test_top_level_package(self) -> None:
        # pylint: disable=import-outside-toplevel
        import test_main
        self.assertIn(test_main.__package__, '', '__package__ should be set to the empty string for top-level modules')


# pylint: disable=unused-argument
def load_tests(loader: TestLoader, standard_tests: TestSuite, pattern: str) -> TestSuite:
    return filter_tests(standard_tests, lambda test_id: 'DelStatement' not in test_id)


def filter_tests(test: TestSuite, filtering_predicate: Callable[[str], bool]) -> TestSuite:
    test_list: list[TestCase | TestSuite] = list(test)
    if len(test_list) == 0:
        return test

    if isinstance(test_list[0], TestCase):
        test_list = list(filter(lambda test_case: filtering_predicate(test_case.id()), test_list))
    else:
        test_list = [filter_tests(test_suit, filtering_predicate) for test_suit in test_list]

    return TestSuite(test_list)
