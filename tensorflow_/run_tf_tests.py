import unittest

if __name__ == '__main__':
    tests = unittest.defaultTestLoader.discover('.', pattern='tf_test*.py')
    unittest.TextTestRunner(verbosity=2).run(tests)
