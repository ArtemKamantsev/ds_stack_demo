from unittest import TestCase
from typing import Union


class Mock:
    p: str | int

    def __init__(self, param: str | int) -> None:
        self.p = param


class ObjectId(TestCase):
    def test_multi_statement(self) -> None:
        obj1: object = object()
        obj2: object = object()
        id1: int = id(obj1)
        id2: int = id(obj2)
        self.assertNotEqual(id1, id2, 'ids should not be the same')

    def test_single_statement(self) -> None:
        id1: int = id(object())
        id2: int = id(object())
        # For some reason objects have identical ids (maybe GC + memory manager do a perfect job?)
        self.assertEqual(id1, id2, 'Ids should be the same')

    def test_single_statement_complex_object(self) -> None:
        id1: int = id(Mock(42))
        id2: int = id(Mock("42"))
        # Ids are identical even then object sizes are different...
        # Maybe ids will be different in case of very significant object sizes difference
        # (memory manager will be forced to look for another part of fee memory)?
        self.assertEqual(id1, id2, 'Ids should be the same')
