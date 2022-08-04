from concurrent.futures import ThreadPoolExecutor
from threading import local, Thread, Lock
from unittest import TestCase


class Threading(TestCase):
    def test_concurrent_increment(self) -> None:
        value: int = 0
        value_lock: Lock = Lock()

        def increment() -> None:
            nonlocal value
            with value_lock:
                value += 1

        number_of_threads: int = 10
        with ThreadPoolExecutor(number_of_threads // 2) as executor:
            for _ in range(number_of_threads):
                executor.submit(increment)

        self.assertEqual(value, number_of_threads)

    def test_attributes_thread_local(self) -> None:
        data: local = local()
        data.value = 42

        def modify_value() -> None:
            data.value = 0

        thread = Thread(target=modify_value)
        thread.start()
        thread.join()

        self.assertEqual(data.value, 42)
