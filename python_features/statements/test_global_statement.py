from unittest import TestCase

__all__ = ['GlobalStatement']

global_value = 0


class GlobalStatement(TestCase):
    def setUp(self) -> None:
        global global_value
        global_value = 0

    def test_access_global_without_global_statement(self):
        self.assertEqual(global_value, 0, 'We can access global variable without "global" statement usage. '
                                          'In this case "global_value" is a free variable')

    def test_modify_global_without_global_statement(self):
        def modify():
            global_value = 42

        modify()
        self.assertNotEqual(global_value, 42, 'We cannot modify global variable without "global" statement usage.'
                                              'Shadowing local variable will be just declared.')

    def test_modify_global_with_global_statement(self):
        def modify():  # (global_value)  # Names listed in a global statement must not be defined as formal parameters
            # v = global_value  # SyntaxError: name 'global_value' is used prior to global declaration
            # global_value = -1  # SyntaxError: name 'global_value' is assigned to before global declaration
            global global_value
            global_value = 42

        modify()
        self.assertEqual(global_value, 42, 'We can modify global variable with "global" statement usage.')

    def test_global_and_nonlocal_usage(self):
        global_value = -1  # shadowing local variable

        def modify():
            global global_value

            def read_global():
                return global_value

            global_value = read_global() + 1

        modify()
        self.assertEqual(global_value, -1, 'Local variables does not affect global statement behaviour')

        temp = global_value

        def read():
            global global_value
            self.assertNotEqual(global_value, temp + 1, 'Local variables does not affect global statement behaviour')

        read()

    def test_create_global_with_global_statement(self):
        def create():
            global global_value_undeclared
            global_value_undeclared = 42

        create()
        self.assertEqual(global_value_undeclared, 42, 'We can declare a global variable with "global" statement usage.')

    def test_modify_global_in_exec(self):
        def modify():
            global global_value
            exec("global_value=42")

        modify()
        self.assertNotEqual(global_value, 42, 'Code contained in a string supplied to "exec" is unaffected by global '
                                              'statements in the code containing the function call')

    def test_modify_global_out_of_exec(self):
        def modify():
            exec("global global_value")
            global_value = 42

        modify()
        self.assertNotEqual(global_value, 42, 'A "global" statement contained in a string or code object supplied to '
                                              'the built-in exec() function does not affect the code block containing '
                                              'the function call')
