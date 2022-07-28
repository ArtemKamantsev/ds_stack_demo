from numbers import Number
from typing import NamedTuple, TypedDict
from unittest import TestCase


class TestMatch(TestCase):
    def test_assignment(self) -> None:
        test_value: tuple[int, int] = (1, 2)
        match test_value:
            case 1, 2 if False:
                self.fail()
            case (2, y) | (1, y) as x:  # only this one should be evaluated
                self.assertEqual(y, 2)
                self.assertEqual(x[1], 2)
            case _:
                self.fail()

    def test_value_pattern(self) -> None:
        class Custom(NamedTuple):
            x: int

        nt_object = Custom(42)

        match 42:
            case nt_object.x as value:
                self.assertEqual(value, 42)
            case _:
                self.fail()

    def test_sequence_pattern(self) -> None:
        test_sequence: list[int] = [1, 2, 3]
        match test_sequence:
            case (1, *rest, 3):
                self.assertEqual(rest, [2])
            case _:
                self.fail()

    def test_mapping_pattern(self) -> None:
        Profile = TypedDict('Profile', {'age': int, 'height': Number})

        test_mapping: Profile = {
            'age': 18,
            'height': 180,
        }
        match test_mapping:
            case {'age': 18, **rest}:
                self.assertEqual(rest, {'height': 180})
            case _:
                self.fail()

    def test_class_pattern(self) -> None:
        ValueHolder = TypedDict('ValueHolder', {'value': int})

        test_mapping: ValueHolder = {
            'value': 42,
        }
        match test_mapping:
            case {'value': int(received_value)}:
                self.assertEqual(received_value, 42)
            case _:
                self.fail()
