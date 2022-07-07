import unittest
from unittest import TestCase, expectedFailure


class ClassScope(TestCase):
    class_variable = 0

    def test_class_variable(self):
        # The scope of names defined in a class block is limited to the class block;
        # it does not extend to the code blocks of methods
        with self.assertRaises(NameError):
            variable = class_variable


if __name__ == '__main__':
    tests = unittest.defaultTestLoader.discover('.')
    unittest.TextTestRunner(verbosity=2).run(tests)
