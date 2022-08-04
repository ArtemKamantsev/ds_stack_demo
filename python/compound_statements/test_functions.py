from unittest import TestCase
from typing import ClassVar


class TestFunction(TestCase):
    def test_default_value(self) -> None:
        class DefaultValue:
            value: int
            value: ClassVar[int] = 0

            def __init__(self):
                # pylint: disable=no-member
                DefaultValue.value += 1
                self.value = DefaultValue.value

        def f(value: DefaultValue = DefaultValue()) -> None:
            # default value is evaluated only once
            self.assertEqual(value.value, 1)

        f()
        f()
