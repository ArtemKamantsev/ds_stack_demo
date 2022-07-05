import unittest
from unittest import TestCase, expectedFailure

from python_features import *


class ClassScope(TestCase):
    class_variable = 0

    def test_class_variable(self):
        # The scope of names defined in a class block is limited to the class block;
        # it does not extend to the code blocks of methods
        with self.assertRaises(NameError):
            variable = class_variable


class Mock:
    def __init__(self, param):
        self.p = param


class ObjectId(TestCase):
    def test_multi_statement(self):
        obj1 = object()
        obj2 = object()
        id1 = id(obj1)
        id2 = id(obj2)
        self.assertNotEqual(id1, id2, 'ids should not be the same')

    def test_single_statement(self):
        id1 = id(object())
        id2 = id(object())
        # For some reason objects have identical ids (maybe GC + memory manager do a perfect job?)
        self.assertEqual(id1, id2, 'Ids should be the same')

    def test_single_statement_complex_object(self):
        id1 = id(Mock(42))
        id2 = id(Mock("42"))
        # Ids are identical even then object sizes are different...
        # Maybe ids will be different in case of very significant object sizes difference
        # (memory manager will be forced to look for another part of fee memory)?
        self.assertEqual(id1, id2, 'Ids should be the same')


def load_tests(loader, standard_tests, pattern):
    test_list = get_custom_test_cases()
    standard_tests.addTests(test_list)

    return standard_tests


if __name__ == '__main__':
    # todo replace by test discovery process
    unittest.main()
