# need this module because for some reason, DataSpell sometimes refuses to run configurations for tests
import unittest

if __name__ == '__main__':
    tests = unittest.defaultTestLoader.discover('.')
    unittest.TextTestRunner(verbosity=2).run(tests)
