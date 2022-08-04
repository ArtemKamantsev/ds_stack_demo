from typing import Any
from unittest import TestCase


class TestSlots(TestCase):
    def test_slots(self) -> None:
        class Common:
            pass

        self.assertIsNotNone(Common().__dict__)

        class CommonSlots:
            __slots__ = []

        with self.assertRaises(AttributeError):
            # pylint: disable=unused-variable
            variables: dict[str, Any] = CommonSlots().__dict__
