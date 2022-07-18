import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from unittest import TestCase


class TestCoroutines(TestCase):
    def test_future(self):
        async def get_value():
            with ThreadPoolExecutor(1) as executor:
                future_value = executor.submit(lambda: 42)  # this future is not awaitable
            value = await asyncio.wrap_future(future_value)
            return value

        result = asyncio.run(get_value())
        self.assertEqual(result, 42)

    def test_double_event_loop(self):
        value = 0

        async def set_value(new_value):
            nonlocal value
            value = new_value

        def run_other_thread():
            asyncio.run(set_value(42))

        async def run():
            with self.assertRaises(RuntimeError):
                asyncio.run(set_value(42))  # one thread allows one event loop
            self.assertEqual(value, 0)

            t = Thread(target=run_other_thread)  # different threads could have own event loops
            t.start()
            t.join()
            self.assertEqual(value, 42)

        asyncio.run(run())

    def test_tasks(self):
        s = set()

        async def put(num):
            s.add(num)

        with self.assertRaises(RuntimeError):
            # there is no event loop
            asyncio.create_task(put(42))

        async def run():
            await asyncio.gather(
                asyncio.create_task(put(0)),
                put(1),  # will be wrapped to task internally
            )

        asyncio.run(run())

        self.assertEqual(s, {0, 1})

    def test_shield(self):
        async def get():
            await asyncio.sleep(0.01)
            return 42

        async def run():
            task1 = asyncio.create_task(get())
            task1.cancel()
            with self.assertRaises(asyncio.exceptions.CancelledError):
                res = await task1

            coro = get()
            task2_shield = asyncio.shield(coro)
            task2_shield.cancel()
            res = await coro
            self.assertEqual(res, 42)

        asyncio.run(run())

    def test_shield_deep(self):
        value = 0

        async def set_value(v):
            nonlocal value
            await asyncio.sleep(0.001)
            value = v

        async def set_executor(task):
            await task

        async def run():
            task1 = asyncio.create_task(set_executor(set_value(42)))
            await asyncio.sleep(0)  # suspend current coroutine and allow another to start execution
            task1.cancel()
            await asyncio.sleep(0.01)
            self.assertEqual(value, 0)  # failed to set value

            task2 = asyncio.create_task(set_executor(asyncio.shield(set_value(42))))
            await asyncio.sleep(0)
            task2.cancel()
            await asyncio.sleep(0.01)
            self.assertEqual(value, 42)  # succeed to set value

        asyncio.run(run())

    def test_shield_wait(self):
        value = 0

        async def set_value(v):
            nonlocal value
            await asyncio.sleep(0.001)
            value = v

        async def run():
            await asyncio.wait_for(set_value(1), 0.1)
            self.assertEqual(value, 1)  # succeed to set value

            with self.assertRaises(asyncio.exceptions.TimeoutError):
                await asyncio.wait_for(set_value(42), 0)
            await asyncio.sleep(0.1)
            self.assertEqual(value, 1)  # failed to set value

            with self.assertRaises(asyncio.exceptions.TimeoutError):
                await asyncio.wait_for(asyncio.shield(set_value(42)), 0)
            await asyncio.sleep(0.1)
            self.assertEqual(value, 42)  # succeed to set value

        asyncio.run(run())

    def test_parallel_computations(self):
        data = []

        class Process:
            def __init__(self, start_value):
                self.start_value = start_value

            def __await__(self):
                data.append(self.start_value)
                yield
                data.append(self.start_value + 1)

                return self.start_value

        async def start_process(start_value):
            return await Process(start_value)

        async def run():
            t1 = asyncio.create_task(start_process(1))  # schedule first process
            t2 = asyncio.create_task(start_process(3))  # schedule second process

            start_value_1 = await t1   # release looper to start working on processes
            start_value_3 = await t2

            self.assertEqual(start_value_1, 1)
            self.assertEqual(start_value_3, 3)
            self.assertEqual(data, [1, 3, 2, 4])  # looper's scheduling is fair, so ordering of array deterministic

        asyncio.run(run())

    def test_custom_awaitable(self):
        class CustomSleep:
            def __init__(self, seconds):
                self.seconds = seconds

            def __await__(self):
                looper = asyncio.get_running_loop()
                f = looper.create_future()
                f._asyncio_future_blocking = True
                # analogy of triggering io event (then looper found any from system in buffer)
                # Future enables integration with low-level callback-based code
                looper.call_later(self.seconds, f.set_result, self.seconds)
                yield f
                # If Task gets Future object yielded, it will resume only then that Future is resolved
                return f.result()

        async def run():
            # Could be modified with 'trampoline' pattern to watch the state of some external source
            min_sleep_time = 0.1
            start_time = time.time()
            res = await CustomSleep(min_sleep_time)
            end_time = time.time()
            self.assertEqual(res, min_sleep_time)
            # looper.call_later guarantees only minimal time elapsed
            self.assertGreater(end_time - start_time, min_sleep_time)

        asyncio.run(run())
