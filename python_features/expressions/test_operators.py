from unittest import TestCase


class TestOperators(TestCase):
    def test_power(self):
        self.assertEqual(-2 ** 2, -4)

    def test_unary(self):
        class CustomUnary:
            def __init__(self, value):
                self.value = value

            def __neg__(self):
                return self.value ** 2

            def __pos__(self):
                return self.value ** 3

            def __invert__(self):
                return -self.value

        self.assertEqual(-CustomUnary(2), 4)
        self.assertEqual(+CustomUnary(2), 8)
        self.assertEqual(~CustomUnary(2), -2)

    def test_sequence_multiplication(self):
        self.assertEqual([1, 2] * 2, [1, 2, 1, 2])

        references_copied = [[1]] * 2
        self.assertIs(references_copied[0], references_copied[1])

        self.assertEqual([1] * -1, [])

    def test_shift_priority(self):
        self.assertEqual(1 << 1 + 1, 4)

    def test_comparison(self):
        self.assertEqual(1 < 3 > 2, True)  # ugly but funny :)

    def test_assignment(self):
        def get_value():
            return 42

        if v := get_value():
            self.assertEqual(v, 42)

    def test_return(self):
        def get_scalar_value():
            v = 0
            try:
                return v
            finally:
                v = 42

        self.assertEqual(get_scalar_value(), 0)

        def get_object_value():
            v = [0]
            try:
                return v
            finally:
                v[0] = 42

        self.assertEqual(get_object_value(), [42])

    def test_raise(self):
        with self.assertRaises(RuntimeError):
            raise  # If there isn’t currently an active exception, a RuntimeError exception is raised

        with self.assertRaises(ArithmeticError):
            try:
                raise ArithmeticError  # object is instantiated using default constructor
            except ArithmeticError:
                raise  # raises currently active exception

        try:
            try:
                raise
            except RuntimeError as e:
                raise ArithmeticError from e  # sets 'e' as '__cause__' of raised ArithmeticError
        except ArithmeticError as e:
            self.assertIs(type(e.__cause__), RuntimeError)

        try:
            try:
                raise
            except RuntimeError:
                raise ArithmeticError  # sets currently active exception as '__context__' of raised ArithmeticError
        except ArithmeticError as e:
            self.assertIs(type(e.__context__), RuntimeError)

    def test_break(self):
        v_scalar = 0
        for i in range(1):
            try:
                break
            finally:
                v_scalar = 42

        self.assertEqual(v_scalar, 42)

    def test_continue(self):
        v_scalar = 0
        for i in range(1):
            try:
                continue
            finally:
                v_scalar = 42

        self.assertEqual(v_scalar, 42)
