from unittest import TestCase


class Mock:
    def __init__(self, param):
        self.p = param


class ObjectId(TestCase):
    def test_multi_statement(self):
        obj1 = object()
        obj2 = object()
        id1 = id(obj1)
        id2 = id(obj2)
        self.assertNotEqual(id1, id2, 'ids should not be the same')

    def test_single_statement(self):
        id1 = id(object())
        id2 = id(object())
        # For some reason objects have identical ids (maybe GC + memory manager do a perfect job?)
        self.assertEqual(id1, id2, 'Ids should be the same')

    def test_single_statement_complex_object(self):
        id1 = id(Mock(42))
        id2 = id(Mock("42"))
        # Ids are identical even then object sizes are different...
        # Maybe ids will be different in case of very significant object sizes difference
        # (memory manager will be forced to look for another part of fee memory)?
        self.assertEqual(id1, id2, 'Ids should be the same')
