from typing import Any
from unittest import TestCase


class TestPrimaries(TestCase):
    def test_custom_attribute_access(self) -> None:
        class Custom:  # just kidding, you should never do so :)
            value: int

            def __init__(self):
                self.value = 0

            def __setattr__(self, name: str, value: int) -> None:
                super().__setattr__(name, value + 1)

            def __getattribute__(self, name: str) -> Any:
                default_value = super().__getattribute__(name)
                if name == 'value':
                    self.value = default_value + 1

                return default_value

        custom: Custom = Custom()
        self.assertEqual(custom.value, 1)
        self.assertEqual(custom.value, 3)
        self.assertEqual(custom.value, 5)
