import unittest
from typing import Callable, Iterable
from unittest import TestCase
from unittest.loader import TestLoader
from unittest.suite import TestSuite


def get_custom_test_cases() -> Iterable[TestSuite | TestCase]:
    def test_function() -> None:
        assert 1 == 1

    def get_comparator(on_failure: type[BaseException]) -> Callable[[object, object, str | None], None]:
        def custom_comparator(m1: object, m2: object, msg: str | None = None) -> None:
            if type(m1) != type(m2):
                raise on_failure(msg)

        return custom_comparator

    class TestClass(TestCase):
        def test_custom_type_equality_function(self) -> None:
            m1: object = object()
            m2: object = object()
            self.assertNotEqual(m1, m2)
            self.addTypeEqualityFunc(object, get_comparator(self.failureException))
            self.assertEqual(m1, m2)

    cases = (
        # can't import FunctionTestCase directly because of test discovery algorithm
        unittest.FunctionTestCase(test_function),
        TestClass('test_custom_type_equality_function'),
    )

    return cases


def load_tests(loader: TestLoader, standard_tests: TestSuite, pattern: str) -> TestSuite:
    test_list: Iterable[TestSuite | TestCase] = get_custom_test_cases()
    standard_tests.addTests(test_list)

    return standard_tests
