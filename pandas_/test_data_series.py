from unittest import TestCase

import pandas as pd
import numpy as np


class TestDataSeries(TestCase):
    def test_non_unique_index(self) -> None:
        data: list[int] = [0, 1, 2]
        series: pd.Series = pd.Series(data, index=[0] * 3)

        self.assertEqual(series.sum(), 3)

    def test_create_from_dict_keys_order(self) -> None:
        data: dict[str, int] = {'c': 2, 'b': 1, 'a': 0}
        series: pd.Series = pd.Series(data)

        # keys' insertion order is preserved if you’re using Python version >= 3.6 and pandas version >= 0.23
        self.assertTrue(isinstance(series.index.to_numpy(), np.ndarray))
        self.assertTrue(np.all(series.index.to_numpy() == np.array(['c', 'b', 'a'])))

    def test_get_missing_key(self) -> None:
        series: pd.Series = pd.Series([0, 1, 2], index=['a', 'b', 'c'])

        with self.assertRaises(KeyError):
            # pylint: disable=unused-variable
            result: int = series['d']

        self.assertIsNone(series.get('d'))
        self.assertTrue(np.isnan(series.get('d', np.nan)))

    def test_label_alignment(self) -> None:
        series: pd.Series = pd.Series(range(2))
        self.assertTrue(np.all(~np.isnan(series)))

        result: pd.Series = series[:1] + series[1:]
        self.assertTrue(np.all(np.isnan(result)))

    def test_name_attribute(self) -> None:
        series_first: pd.Series = pd.Series(range(2), name='first')
        series_second: pd.Series = series_first.rename('second')

        self.assertFalse(series_first is series_second)
        self.assertTrue(np.all(series_first == series_second))

    def test_default_iteration(self) -> None:
        data: np.ndarray = np.zeros((2,), dtype=int)
        series: pd.Series = pd.Series(data, index=['A', 'B'])

        self.assertTrue((np.asarray(series) == [0, 0]).all())

    def test_dt_accessor(self) -> None:
        series: pd.Series = pd.Series(pd.date_range("20220101 12:00:00", periods=4))

        self.assertTrue((series.dt.day.to_numpy() == [1, 2, 3, 4]).all())

    def test_str_accessor(self) -> None:
        series: pd.Series = pd.Series(['A', 'B', 'C'])
        result: pd.Series = series.str.lower()

        self.assertTrue((result == ['a', 'b', 'c']).all())

    def test_search_sorted(self) -> None:
        series: pd.Series = pd.Series([0, 1, 3])
        result: np.ndarray = series.searchsorted([2, 4])

        # pylint: disable=no-member
        self.assertTrue((result == [2, 3]).all())

    def test_sparce_accessor(self) -> None:
        series: pd.Series = pd.Series([0, 0, 1, 2], dtype="Sparse[int]")

        self.assertAlmostEqual(series.sparse.density, 0.5, 1, delta=0.01)

    def test_nullable_boolean_data_type(self) -> None:
        result1: pd.Series = pd.Series([True, False, np.nan], dtype="object") | True
        result2: pd.Series = pd.Series([True, False, np.nan], dtype="boolean") | True

        self.assertTrue((result1 == [True, True, False]).all())
        self.assertTrue(result2.all())
