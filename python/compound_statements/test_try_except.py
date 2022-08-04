import sys
from types import TracebackType
from unittest import TestCase


class TestTryExcept(TestCase):
    def test_except_name(self) -> None:
        try:
            # pylint: disable=misplaced-bare-raise
            raise
        # pylint: disable=unused-variable
        except RuntimeError as error:
            pass

        with self.assertRaises(NameError):
            # pylint: disable=used-before-assignment, unused-variable
            error_local = error  # the name is cleared

    def test_access_exception(self) -> None:
        error_type: type[RuntimeError]
        error: RuntimeError
        traceback: TracebackType

        try:
            # pylint: disable=misplaced-bare-raise
            raise
        except RuntimeError:
            error_type, error, traceback = sys.exc_info()
            self.assertEqual(error_type, RuntimeError)
            self.assertEqual(type(error), error_type)
        finally:
            # The exception information is not available to the program during execution of the finally clause
            error_type, error, traceback = sys.exc_info()
            self.assertIsNone(error_type)
            self.assertIsNone(error)
            self.assertIsNone(traceback)

    def test_return_override_value(self) -> None:
        def get_value() -> int:
            # pylint: disable=lost-exception
            try:
                return 0
            finally:

                return 42

        self.assertEqual(get_value(), 42)

    def test_return_exception(self) -> None:
        def get_value() -> int:
            # pylint: disable=lost-exception
            try:
                # pylint: disable=misplaced-bare-raise
                raise
            finally:
                return 42

        self.assertEqual(get_value(), 42)
