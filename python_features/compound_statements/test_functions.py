from unittest import TestCase
from typing import ClassVar


class TestFunction(TestCase):
    def test_default_value(self) -> None:
        class DefaultValue:
            value: ClassVar[int] = 0
            value: int

            def __init__(self):
                DefaultValue.value += 1
                self.value = DefaultValue.value

        def f(value: DefaultValue = DefaultValue()) -> None:
            # default value is evaluated only once
            self.assertEqual(value.value, 1)

        f()
        f()
