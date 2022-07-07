import unittest
from unittest import TestCase


def get_custom_test_cases():
    def test_function():
        assert 1 == 1

    def get_comparator(on_failure):
        def custom_comparator(m1, m2, msg=None):
            if type(m1) != type(m2):
                raise on_failure(msg)

        return custom_comparator

    class TestClass(TestCase):
        def test_custom_type_equality_function(self):
            m1 = object()
            m2 = object()
            self.assertNotEqual(m1, m2)
            self.addTypeEqualityFunc(object, get_comparator(self.failureException))
            self.assertEqual(m1, m2)

    cases = (
        # can't import FunctionTestCase directly because of test discovery algorithm
        unittest.FunctionTestCase(test_function),
        TestClass('test_custom_type_equality_function'),
    )

    return cases


def load_tests(loader, standard_tests, pattern):
    test_list = get_custom_test_cases()
    standard_tests.addTests(test_list)

    return standard_tests
