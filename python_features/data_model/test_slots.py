from unittest import TestCase


class TestSlots(TestCase):
    def test_slots(self):
        class Common:
            pass

        self.assertIsNotNone(Common().__dict__)

        class CommonSlots:
            __slots__ = []

        with self.assertRaises(AttributeError):
            d = CommonSlots().__dict__
