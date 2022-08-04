from unittest import TestCase

__all__ = ['GlobalStatement']

GLOBAL_VALUE: int = 0


class GlobalStatement(TestCase):
    def setUp(self) -> None:
        # pylint: disable=global-statement
        global GLOBAL_VALUE
        GLOBAL_VALUE = 0

    def test_access_global_without_global_statement(self) -> None:
        self.assertEqual(GLOBAL_VALUE, 0, 'We can access global variable without "global" statement usage. '
                                          'In this case "GLOBAL_VALUE" is a free variable')

    def test_modify_global_without_global_statement(self) -> None:
        def modify() -> None:
            # pylint: disable=redefined-outer-name, invalid-name, unused-variable
            GLOBAL_VALUE = 42

        modify()
        self.assertNotEqual(GLOBAL_VALUE, 42, 'We cannot modify global variable without "global" statement usage.'
                                              'Shadowing local variable will be just declared.')

    def test_modify_global_with_global_statement(self) -> None:
        # def modify(GLOBAL_VALUE)  # Names listed in a global statement must not be defined as formal parameters
        def modify() -> None:
            # v = GLOBAL_VALUE  # SyntaxError: name 'GLOBAL_VALUE' is used prior to global declaration
            # GLOBAL_VALUE = -1  # SyntaxError: name 'GLOBAL_VALUE' is assigned to before global declaration
            # pylint: disable=global-statement
            global GLOBAL_VALUE
            GLOBAL_VALUE = 42

        modify()
        self.assertEqual(GLOBAL_VALUE, 42, 'We can modify global variable with "global" statement usage.')

    def test_global_and_nonlocal_usage(self) -> None:
        # pylint: disable=redefined-outer-name, invalid-name
        GLOBAL_VALUE: int = -1  # shadowing local variable

        def modify() -> None:
            # pylint: disable=global-statement
            global GLOBAL_VALUE

            def read_global() -> int:
                return GLOBAL_VALUE

            GLOBAL_VALUE = read_global() + 1

        modify()
        self.assertEqual(GLOBAL_VALUE, -1, 'Local variables does not affect global statement behaviour')

        temp: int = GLOBAL_VALUE

        def read() -> None:
            # pylint: disable=global-variable-not-assigned
            global GLOBAL_VALUE
            self.assertNotEqual(GLOBAL_VALUE, temp + 1, 'Local variables does not affect global statement behaviour')

        read()

    def test_create_global_with_global_statement(self) -> None:
        def create() -> None:
            # pylint: disable=invalid-name, global-variable-not-assigned, global-variable-undefined
            global global_value_undeclared
            global_value_undeclared = 42

        create()
        self.assertEqual(global_value_undeclared, 42, 'We can declare a global variable with "global" statement usage.')

    def test_modify_global_in_exec(self) -> None:
        def modify() -> None:
            # pylint: disable=global-variable-not-assigned, exec-used
            global GLOBAL_VALUE
            exec("GLOBAL_VALUE=42")

        modify()
        self.assertNotEqual(GLOBAL_VALUE, 42, 'Code contained in a string supplied to "exec" is unaffected by global '
                                              'statements in the code containing the function call')

    def test_modify_global_out_of_exec(self) -> None:
        def modify() -> None:
            # pylint: disable=exec-used, redefined-outer-name, invalid-name, unused-variable
            exec("global GLOBAL_VALUE")
            GLOBAL_VALUE = 42

        modify()
        self.assertNotEqual(GLOBAL_VALUE, 42, 'A "global" statement contained in a string or code object supplied to '
                                              'the built-in exec() function does not affect the code block containing '
                                              'the function call')
