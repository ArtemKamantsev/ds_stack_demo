import threading
from unittest import TestCase


class ThreadingModuleFunctions(TestCase):
    def test_main_thread(self):
        self.assertEqual(threading.main_thread(), threading.current_thread(),
                         'The test should be run from the main thread!')

    def test_thread_ident(self):
        thread = threading.Thread(target=lambda: 42)
        self.assertIsNone(thread.ident)
        thread.start()
        thread.join()
        self.assertIsNotNone(thread.ident)
