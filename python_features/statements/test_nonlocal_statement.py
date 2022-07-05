from unittest import TestCase

__all__ = ['NonlocalStatement']

global_value = 0


class NonlocalStatement(TestCase):
    def setUp(self) -> None:
        global global_value
        global_value = 0

    def test_access_nonlocal_without_nonlocal_statement(self):
        local_value = 0

        def read():
            return local_value

        self.assertEqual(local_value, read(), 'We can access nonlocal variable without "nonlocal" statement usage. '
                                              'In this case "local_value" is a free variable')

    def test_redefine_free_variable(self):
        free_value = 0

        def read():
            value = free_value  # local variable 'free_value' referenced before assignment
            free_value = value
            return free_value

        with self.assertRaises(UnboundLocalError):
            read()

    def test_modify_nonlocal_without_nonlocal_statement(self):
        local_value = 0

        def modify():
            local_value = 42

        modify()
        self.assertEqual(local_value, 0, 'We cannot modify nonlocal variable without "nonlocal" statement usage.'
                                         'Shadowing local variable will be just declared.')

    def test_modify_nonlocal_with_nonlocal_statement(self):
        local_value = 0

        def modify():  # (local_value)  # Names listed in a nonlocal statement must not be defined as formal parameters
            # v = local_value  # SyntaxError: name 'local_value' is used prior to nonlocal declaration
            # local_value = -1 # SyntaxError: name 'local_value' is assigned to before nonlocal declaration
            # nonlocal global_value  # SyntaxError: no binding for nonlocal 'global_value' found
            nonlocal local_value
            local_value = 42

        modify()
        self.assertEqual(local_value, 42, 'We can modify nonlocal variable with "nonlocal" statement usage.')

    def test_modify_multiple_scopes_single_nonlocal(self):
        local_value = 0

        def modify_lvl1():
            def modify_lvl2():
                nonlocal local_value
                local_value = 42

            modify_lvl2()

        modify_lvl1()
        self.assertEqual(local_value, 42, 'We can modify nonlocal variable with "nonlocal" statement usage '
                                          'even a few level above.')

    def test_modify_multiple_scopes_multiple_nonlocal(self):
        local_value = 0

        def modify_lvl1():
            local_value = -1

            def modify_lvl2():
                nonlocal local_value
                local_value = 42

            modify_lvl2()

        modify_lvl1()
        self.assertEqual(local_value, 0, 'We can modify nonlocal variable with "nonlocal" statement usage '
                                         'only from the nearest enclosing scope')

    def test_create_nonlocal_with_nonlocal_statement(self):
        # def create():
        #     nonlocal local_value    # SyntaxError: no binding for nonlocal 'local_value' found (the scope in which a
        #                             # new binding should be created
        #                             # cannot be determined unambiguously)
        #     local_value = 42
        #
        # create()
        pass

    def test_read_free_variable_in_exec_without_context(self):
        local_variable = 0

        def read():
            exec("value = local_variable")

        with self.assertRaises(NameError):
            read()

    def test_read_free_variable_in_exec_with_context(self):
        local_variable = 0

        def read():
            exec("value = local_variable", {'local_variable': local_variable})

        self.assertIsNone(read())
