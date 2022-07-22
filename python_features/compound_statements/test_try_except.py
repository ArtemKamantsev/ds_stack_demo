import sys
from unittest import TestCase


class TestTryExcept(TestCase):
    def test_except_name(self):
        try:
            raise
        except RuntimeError as e:
            pass

        with self.assertRaises(NameError):
            ex = e  # the name is cleared

    def test_access_exception(self):
        try:
            raise
        except:
            error_type, error, traceback = sys.exc_info()
            self.assertEqual(error_type, RuntimeError)
            self.assertEqual(type(error), error_type)
        finally:
            # The exception information is not available to the program during execution of the finally clause
            error_type, error, traceback = sys.exc_info()
            self.assertIsNone(error_type)
            self.assertIsNone(error)
            self.assertIsNone(traceback)

    def test_return_override(self):
        def get_value():
            try:
                return 0
            finally:
                return 42

        self.assertEqual(get_value(), 42)
