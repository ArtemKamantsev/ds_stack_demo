from unittest import TestCase

__all__ = ['NonlocalStatement']

GLOBAL_VALUE: int = 0


class NonlocalStatement(TestCase):
    def setUp(self) -> None:
        # pylint: disable=global-statement
        global GLOBAL_VALUE
        GLOBAL_VALUE = 0

    def test_access_nonlocal_without_nonlocal_statement(self) -> None:
        local_value: int = 0

        def read() -> int:
            return local_value

        self.assertEqual(local_value, read(), 'We can access nonlocal variable without "nonlocal" statement usage. '
                                              'In this case "local_value" is a free variable')

    def test_redefine_free_variable(self) -> None:
        # pylint: disable=unused-variable
        free_value: int = 0

        def read() -> int:
            # pylint: disable=used-before-assignment
            value: int = free_value  # local variable 'free_value' referenced before assignment
            free_value: int = value

            return free_value

        with self.assertRaises(UnboundLocalError):
            read()

    def test_modify_nonlocal_without_nonlocal_statement(self) -> None:
        local_value: int = 0

        def modify() -> None:
            # pylint: disable=unused-variable
            local_value = 42

        modify()
        self.assertEqual(local_value, 0, 'We cannot modify nonlocal variable without "nonlocal" statement usage.'
                                         'Shadowing local variable will be just declared.')

    def test_modify_nonlocal_with_nonlocal_statement(self) -> None:
        local_value: int = 0

        # def modify(local_value)  # Names listed in a nonlocal statement must not be defined as formal parameters
        def modify() -> None:
            # v = local_value  # SyntaxError: name 'local_value' is used prior to nonlocal declaration
            # local_value = -1 # SyntaxError: name 'local_value' is assigned to before nonlocal declaration
            # nonlocal GLOBAL_VALUE  # SyntaxError: no binding for nonlocal 'GLOBAL_VALUE' found
            nonlocal local_value
            local_value = 42

        modify()
        self.assertEqual(local_value, 42, 'We can modify nonlocal variable with "nonlocal" statement usage.')

    def test_modify_multiple_scopes_single_nonlocal(self) -> None:
        local_value: int = 0

        def modify_lvl1() -> None:
            def modify_lvl2() -> None:
                nonlocal local_value
                local_value = 42

            modify_lvl2()

        modify_lvl1()
        self.assertEqual(local_value, 42, 'We can modify nonlocal variable with "nonlocal" statement usage '
                                          'even a few level above.')

    def test_modify_multiple_scopes_multiple_nonlocal(self) -> None:
        local_value: int = 0

        def modify_lvl1() -> None:
            local_value: int = -1

            def modify_lvl2():
                nonlocal local_value
                local_value = 42

            modify_lvl2()

        modify_lvl1()
        self.assertEqual(local_value, 0, 'We can modify nonlocal variable with "nonlocal" statement usage '
                                         'only from the nearest enclosing scope')

    def test_create_nonlocal_with_nonlocal_statement(self) -> None:
        # def create():
        #     nonlocal local_value    # SyntaxError: no binding for nonlocal 'local_value' found (the scope in which a
        #                             # new binding should be created
        #                             # cannot be determined unambiguously)
        #     local_value = 42
        #
        # create()
        pass

    def test_read_free_variable_in_exec_without_context(self) -> None:
        # pylint: disable=unused-variable
        local_variable: int = 0

        def read() -> None:
            # pylint: disable=exec-used
            exec("value = local_variable")

        with self.assertRaises(NameError):
            read()

    def test_read_free_variable_in_exec_with_context(self) -> None:
        local_variable: int = 0

        def read() -> None:
            # pylint: disable=exec-used
            exec("value = local_variable", {'local_variable': local_variable})

        self.assertIsNone(read())
