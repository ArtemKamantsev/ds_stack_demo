from multiprocessing import Process, Queue, Pipe, Lock, Value
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
from concurrent.futures import ProcessPoolExecutor
from unittest import TestCase


def _queue_child_process(q):
    q.put(42)


def _pipe_child_process(conn):
    conn.send(42)
    conn.close()


def _increment(variable):
    variable.value = variable.value + 1


def _increment_shared_memory(name, lock):
    memory = SharedMemory(name, create=False)
    with lock:
        memory.buf[0] += 1
    memory.close()


def _increment_shared_memory_at(data, idx):
    data[idx] += 1


class DataExchange(TestCase):

    def test_queue(self):
        queue = Queue()

        p = Process(target=_queue_child_process, args=(queue,))
        p.start()
        self.assertEqual(queue.get(), 42)
        p.join()

    def test_pipe(self):
        parent_conn, child_conn = Pipe()

        p = Process(target=_pipe_child_process, args=(child_conn,))
        p.start()
        self.assertEqual(parent_conn.recv(), 42)
        p.join()

    def test_shared_value(self):
        num = Value('d', 0)

        number_of_processes = 2
        processes = [Process(target=_increment, args=(num,)) for _ in range(number_of_processes)]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

        self.assertEqual(num.value, number_of_processes)

    def test_shared_memory(self):
        name = 'test'
        memory = SharedMemory(name, create=True, size=1)
        memory_lock = Lock()
        memory.buf[0] = 0

        number_of_processes = 2
        processes = [Process(target=_increment_shared_memory, args=(name, memory_lock))
                     for _ in range(number_of_processes)]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

        self.assertEqual(memory.buf[0], number_of_processes)
        memory.close()
        memory.unlink()  # should be called only once

    def test_shared_memory_manager(self):
        number_of_processes = 2
        with SharedMemoryManager() as smm:
            data = smm.ShareableList([0] * number_of_processes)

            processes = [Process(target=_increment_shared_memory_at, args=(data, i))
                         for i in range(number_of_processes)]
            for p in processes:
                p.start()
            for p in processes:
                p.join()

            self.assertEqual(sum(data), number_of_processes)
        # no cleanup required