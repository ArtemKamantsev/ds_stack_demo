from unittest import TestCase


class TestFunction(TestCase):
    def test_default_value(self):
        class DefaultValue:
            value = 0

            def __init__(self):
                DefaultValue.value += 1
                self.value = DefaultValue.value

        def f(value=DefaultValue()):
            # default value is evaluated only once
            self.assertEqual(value.value, 1)

        f()
        f()
