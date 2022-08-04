import unittest
from typing import Callable, Iterable
from unittest import TestCase
from unittest.loader import TestLoader
from unittest.suite import TestSuite


def get_custom_test_cases() -> Iterable[TestSuite | TestCase]:
    def test_function() -> None:
        # pylint: disable=comparison-with-itself, comparison-of-constants
        assert 1 == 1

    def get_comparator(on_failure: type[BaseException]) -> Callable[[object, object, str | None], None]:
        def custom_comparator(first: object, second: object, msg: str | None = None) -> None:
            if not isinstance(first, type(second)):
                raise on_failure(msg)

        return custom_comparator

    class TestClass(TestCase):
        def test_custom_type_equality_function(self) -> None:
            first: object = object()
            second: object = object()
            self.assertNotEqual(first, second)
            self.addTypeEqualityFunc(object, get_comparator(self.failureException))
            self.assertEqual(first, second)

    cases = (
        # can't import FunctionTestCase directly because of test discovery algorithm
        unittest.FunctionTestCase(test_function),
        TestClass('test_custom_type_equality_function'),
    )

    return cases


# pylint: disable=unused-argument
def load_tests(loader: TestLoader, standard_tests: TestSuite, pattern: str) -> TestSuite:
    test_list: Iterable[TestSuite | TestCase] = get_custom_test_cases()
    standard_tests.addTests(test_list)

    return standard_tests
