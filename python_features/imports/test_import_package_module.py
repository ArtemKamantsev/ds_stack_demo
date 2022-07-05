from unittest import TestCase

from python_features.statements.test_del_statement import *

__all__ = ['Import']

temp = (DelStatement,)  # just to keep import during imports optimization operation


class Import(TestCase):
    def test_redundant_imports(self):
        with self.assertRaises(NameError):
            global_value  # import avoided by using '__all__' attribute + import * syntax

    def test_path(self):
        import python_features as package
        import python_features.statements.test_del_statement as module

        self.assertIsNotNone(package.__path__, 'Packages - are modules and has a __path__ attribute')
        # Non package modules do not have a __path__ attribute
        with self.assertRaises(AttributeError):
            module.__path__

    def test_name_package(self):
        import python_features.statements
        import python_features.statements.test_del_statement as module

        self.assertEqual(python_features.__name__, python_features.__package__, 'The name of the top level package '
                                                                                'that contains the module is bound '
                                                                                'in the local namespace as a reference '
                                                                                'to the top level package.')
        self.assertNotEqual(module.__name__, module.__package__)
        self.assertEqual(module.__package__, python_features.statements.__name__, 'Fully qualified name'
                                                                                  'must be used because no alias'
                                                                                  'was used')

    def test_top_level_package(self):
        import main
        self.assertIn(main.__package__, '', '__package__ should be set to the empty string for top-level modules')
