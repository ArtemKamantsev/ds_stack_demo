from numbers import Number
from typing import Callable, ParamSpec, TypeVar
from unittest import TestCase


class TestDecorators(TestCase):
    def test_function_decorator(self) -> None:
        # pylint: disable=invalid-name
        P = ParamSpec('P')

        def multiply(factor: Number) -> Callable[[Callable[[P], Number]], Callable[[P], Number]]:
            def decorator(f: Callable[[P], Number]) -> Callable[[P], Number]:
                def modified_f(*args: P.args, **kwargs: P.kwargs) -> Number:
                    return f(*args, **kwargs) * factor

                return modified_f

            return decorator

        @multiply(2)
        def get_value() -> int:
            return 42

        self.assertEqual(get_value(), 42 * 2)

    def test_class_decorator(self) -> None:
        # pylint: disable=invalid-name
        DesiredType = TypeVar('DesiredType')

        def replace_type(desired_type: DesiredType) -> Callable[[type], DesiredType]:
            return lambda given_type: desired_type

        class A:
            value: int

            def __init__(self):
                self.value = 42

        class B:
            value: int

            def __init__(self):
                self.value = 0

        B = A

        @replace_type(A)
        class C:
            value: int

            def __init__(self):
                self.value = 1

        b: B = B()  # actually, 'B' and 'C' types aren't present anymore, these names just refer to type 'A'
        c: C = C()
        self.assertEqual(b.value, 42)
        self.assertIsInstance(b, A)
        self.assertIs(B, A)

        self.assertEqual(c.value, 42)
        self.assertIsInstance(c, A)
        self.assertIs(C, A)
