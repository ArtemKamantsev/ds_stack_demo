from typing import Any
from unittest import TestCase


class TestPrimaries(TestCase):
    def test_custom_attribute_access(self) -> None:
        class Custom:  # just kidding, you should never do so :)
            a: int

            def __init__(self):
                self.a = 0

            def __setattr__(self, name: str, value: int) -> None:
                super().__setattr__(name, value + 1)

            def __getattribute__(self, name: str) -> Any:
                default_value = super().__getattribute__(name)
                if name == 'a':
                    self.a = default_value + 1

                return default_value

        c: Custom = Custom()
        self.assertEqual(c.a, 1)
        self.assertEqual(c.a, 3)
        self.assertEqual(c.a, 5)
