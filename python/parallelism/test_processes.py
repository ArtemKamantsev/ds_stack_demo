from multiprocessing import Process, Queue, Pipe, Lock, Value
from multiprocessing.connection import Connection
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory, ShareableList
from unittest import TestCase


def _queue_child_process(queue: Queue) -> None:
    queue.put(42)


def _pipe_child_process(conn: Connection) -> None:
    conn.send(42)
    conn.close()


def _increment(variable: Value) -> None:
    variable.value = variable.value + 1


def _increment_shared_memory(name: str, lock: Lock) -> None:
    memory: SharedMemory = SharedMemory(name, create=False)
    with lock:
        memory.buf[0] += 1
    memory.close()


def _increment_shared_memory_at(data: ShareableList, idx: int) -> None:
    data[idx] += 1


class TestProcesses(TestCase):

    def test_queue(self) -> None:
        queue: Queue = Queue()

        process: Process = Process(target=_queue_child_process, args=(queue,))
        process.start()
        self.assertEqual(queue.get(), 42)
        process.join()

    def test_pipe(self) -> None:
        child_conn: Connection
        child_conn: Connection
        parent_conn, child_conn = Pipe()

        process: Process = Process(target=_pipe_child_process, args=(child_conn,))
        process.start()
        self.assertEqual(parent_conn.recv(), 42)
        process.join()

    def test_shared_value(self) -> None:
        num: Value = Value('d', 0)

        number_of_processes: int = 2
        processes: tuple[Process] = tuple(Process(target=_increment, args=(num,)) for _ in range(number_of_processes))
        for process in processes:
            process.start()
        for process in processes:
            process.join()

        self.assertEqual(num.value, number_of_processes)

    def test_shared_memory(self) -> None:
        name: str = 'test'
        memory: SharedMemory = SharedMemory(name, create=True, size=1)
        memory_lock: Lock = Lock()
        memory.buf[0] = 0

        number_of_processes: int = 2
        processes: tuple[Process] = tuple(Process(target=_increment_shared_memory, args=(name, memory_lock))
                                          for _ in range(number_of_processes))
        for process in processes:
            process.start()
        for process in processes:
            process.join()

        self.assertEqual(memory.buf[0], number_of_processes)
        memory.close()
        memory.unlink()  # should be called only once

    def test_shared_memory_manager(self) -> None:
        number_of_processes: int = 2
        with SharedMemoryManager() as smm:
            data: ShareableList = smm.ShareableList([0] * number_of_processes)

            processes: tuple[Process] = tuple(Process(target=_increment_shared_memory_at, args=(data, i))
                                              for i in range(number_of_processes))
            for process in processes:
                process.start()
            for process in processes:
                process.join()

            self.assertEqual(sum(data), number_of_processes)
        # no cleanup required
