from unittest import TestCase


class TestDecorators(TestCase):
    def test_function_decorator(self):
        def multiply(factor):
            def decorator(f):
                return lambda *args, **kwargs: f(*args, **kwargs) * factor

            return decorator

        @multiply(2)
        def get_value():
            return 42

        self.assertEqual(get_value(), 84)

    def test_class_decorator(self):
        def replace_type(desired_type):
            return lambda given_type: desired_type

        class A:
            def __init__(self):
                self.value = 42

        class B:
            def __init__(self):
                self.value = 0

        B = A

        @replace_type(A)
        class C:
            def __init__(self):
                self.value = 1

        b = B()  # actually, 'B' and 'C' types aren't present anymore, these names just refer to type 'A'
        c = C()
        self.assertEqual(b.value, 42)
        self.assertIsInstance(b, A)
        self.assertIs(B, A)

        self.assertEqual(c.value, 42)
        self.assertIsInstance(c, A)
        self.assertIs(C, A)
