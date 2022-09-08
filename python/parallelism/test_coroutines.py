import asyncio
import time
import warnings
from asyncio import Task, Future
from collections.abc import Coroutine, Awaitable
from concurrent.futures import ThreadPoolExecutor, Future as ConcurrentFuture
from threading import Thread
from typing import Any
from unittest import TestCase


class TestCoroutines(TestCase):
    def test_future(self) -> None:
        async def get_value() -> int:
            with ThreadPoolExecutor(1) as executor:
                future_value: ConcurrentFuture[int] = executor.submit(lambda: 42)  # this future is not awaitable
            value: int = await asyncio.wrap_future(future_value)
            return value

        result: int = asyncio.run(get_value())
        self.assertEqual(result, 42)

    def test_double_event_loop(self) -> None:
        warnings.filterwarnings('ignore',
                                r'coroutine \'TestCoroutines\.test_double_event_loop\.<locals>\.set_value\' '
                                r'was never awaited',
                                category=RuntimeWarning)
        value: int = 0

        async def set_value(new_value: int) -> None:
            nonlocal value
            value = new_value

        def run_other_thread() -> None:
            asyncio.run(set_value(42))

        async def run() -> None:
            with self.assertRaises(RuntimeError):
                asyncio.run(set_value(42))  # one thread allows one event loop
            self.assertEqual(value, 0)

            thread: Thread = Thread(target=run_other_thread)  # different threads could have own event loops
            thread.start()
            thread.join()
            self.assertEqual(value, 42)

        asyncio.run(run())

    def test_tasks(self) -> None:
        data: set[int] = set()

        async def put(num: int) -> None:
            data.add(num)

        with self.assertRaises(RuntimeError):
            warnings.filterwarnings('ignore',
                                    r'coroutine \'TestCoroutines.test_tasks\.<locals>\.put\' was never awaited',
                                    category=RuntimeWarning)
            # there is no event loop
            asyncio.create_task(put(42))

        async def run() -> None:
            await asyncio.gather(
                    asyncio.create_task(put(0)),
                    put(1),  # will be wrapped to task internally
            )

        asyncio.run(run())

        self.assertEqual(data, {0, 1})

    def test_shield(self) -> None:
        async def get() -> int:
            await asyncio.sleep(0.01)
            return 42

        async def run() -> None:
            task1: Task[int] = asyncio.create_task(get())
            task1.cancel()
            with self.assertRaises(asyncio.exceptions.CancelledError):
                res: int = await task1

            coro: Coroutine[Any, Any, int] = get()
            task2_shield: Future[int] = asyncio.shield(coro)
            task2_shield.cancel()
            # raises exception in task2_shield, todo: handle this exception somehow to remove warning message
            res: int = await coro
            self.assertEqual(res, 42)

        asyncio.run(run())

    def test_shield_deep(self) -> None:
        value: int = 0

        async def set_value(arg_value: int) -> None:
            nonlocal value
            await asyncio.sleep(0.001)
            value = arg_value

        async def set_executor(task: Awaitable) -> None:
            await task

        async def run() -> None:
            task1: Task[None] = asyncio.create_task(set_executor(set_value(42)))
            await asyncio.sleep(0)  # suspend current coroutine and allow another to start execution
            task1.cancel()
            await asyncio.sleep(0.01)
            self.assertEqual(value, 0)  # failed to set value

            task2: Task[None] = asyncio.create_task(set_executor(asyncio.shield(set_value(42))))
            await asyncio.sleep(0)
            task2.cancel()
            await asyncio.sleep(0.01)
            self.assertEqual(value, 42)  # succeed to set value

        asyncio.run(run())

    def test_shield_wait(self) -> None:
        value: int = 0

        async def set_value(arg_value: int) -> None:
            nonlocal value
            await asyncio.sleep(0.001)
            value = arg_value

        async def run() -> None:
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

    def test_parallel_computations(self) -> None:
        data: list[int] = []

        class Process:
            start_value: int

            def __init__(self, start_value: int):
                self.start_value = start_value

            def __await__(self) -> int:
                data.append(self.start_value)
                yield
                data.append(self.start_value + 1)

                return self.start_value

        async def start_process(start_value: int) -> int:
            return await Process(start_value)

        async def run():
            task1: Task[int] = asyncio.create_task(start_process(1))  # schedule first process
            task2: Task[int] = asyncio.create_task(start_process(3))  # schedule second process

            start_value_1: int = await task1  # release looper to start working on processes
            start_value_3: int = await task2

            self.assertEqual(start_value_1, 1)
            self.assertEqual(start_value_3, 3)
            self.assertEqual(data, [1, 3, 2, 4])  # looper's scheduling is fair, so ordering of array deterministic

        asyncio.run(run())

    def test_custom_awaitable(self) -> None:
        class CustomSleep:
            seconds: float

            def __init__(self, seconds: float | int):
                self.seconds = seconds

            def __await__(self) -> float:
                looper = asyncio.get_running_loop()
                f: Future[Any] = looper.create_future()
                f._asyncio_future_blocking = True
                # analogy of triggering io event (then looper found any from system in buffer)
                # Future enables integration with low-level callback-based code
                looper.call_later(self.seconds, f.set_result, self.seconds)
                yield f
                # If Task gets Future object yielded, it will resume only then that Future is resolved
                return f.result()

        async def run() -> None:
            # Could be modified with 'trampoline' pattern to watch the state of some external source
            min_sleep_time: float = 0.1
            start_time: float = time.time()
            res: float = await CustomSleep(min_sleep_time)
            end_time: float = time.time()
            self.assertEqual(res, min_sleep_time)
            # looper.call_later guarantees only minimal time elapsed
            self.assertGreater(end_time - start_time, min_sleep_time)

        asyncio.run(run())
