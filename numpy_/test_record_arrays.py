from unittest import TestCase

import numpy as np


class TestRecordArrays(TestCase):
    def test_record_array_fields_access(self) -> None:
        data: np.recarray = np.rec.array([(180, ('Jim', 'Baily')),
                                          (160, ('Kate', 'Raily'))], dtype=[('height', float),
                                                                            ('name',
                                                                             [('first', 'U10'), ('last', 'U10')])])
        self.assertEqual(data[1].height, 160.0)
        # fields accessed by index or by attribute are returned as a ndarray if the field has not a structured type
        self.assertTrue(np.all(data.height == np.array([180, 160], dtype=float)))
        self.assertTrue(np.all(data[1:].height == np.array([160], dtype=float)))
        self.assertEqual(data[1].name.last, 'Raily')
        # fields accessed by index or by attribute are returned as a record array if the field has a structured type
        self.assertTrue(isinstance(data.name, np.recarray))
