from collections import namedtuple
from unittest import TestCase


class TestMatch(TestCase):
    def test_assignment(self):
        test_value = (1, 2)
        match test_value:
            case 1, 2 if False:
                self.fail()
            case (2, y) | (1, y) as x:  # only this one should be evaluated
                self.assertEqual(y, 2)
                self.assertEqual(x[1], 2)
            case _:
                self.fail()

    def test_value_pattern(self):
        nt_class = namedtuple('custom', 'x')
        nt_object = nt_class(42)

        match 42:
            case nt_object.x as value:
                self.assertEqual(value, 42)
            case _:
                self.fail()

    def test_sequence_pattern(self):
        test_sequence = [1, 2, 3]
        match test_sequence:
            case (1, *rest, 3):
                self.assertEqual(rest, [2])
            case _:
                self.fail()

    def test_mapping_pattern(self):
        test_mapping = {
            'age': 18,
            'height': 180,
        }
        match test_mapping:
            case {'age': 18, **rest}:
                self.assertEqual(rest, {'height': 180})
            case _:
                self.fail()

    def test_class_pattern(self):
        test_mapping = {
            'value': 42,
        }
        match test_mapping:
            case {'value': int(received_value)}:
                self.assertEqual(received_value, 42)
            case _:
                self.fail()
