from concurrent.futures import ThreadPoolExecutor
from threading import local, Thread, Lock
from unittest import TestCase


class Threading(TestCase):
    def test_concurrent_increment(self):
        value = 0
        value_lock = Lock()

        def increment():
            nonlocal value
            with value_lock:
                value += 1

        number_of_threads = 10
        with ThreadPoolExecutor(number_of_threads // 2) as executor:
            for _ in range(number_of_threads):
                executor.submit(increment)

        self.assertEqual(value, number_of_threads)

    def test_attributes_thread_local(self):
        data = local()
        data.value = 42

        def modify_value():
            data.value = 0

        thread = Thread(target=modify_value)
        thread.start()
        thread.join()

        self.assertEqual(data.value, 42)
