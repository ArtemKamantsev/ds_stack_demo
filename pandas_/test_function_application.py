from functools import partial
from numbers import Number
from typing import Callable
from unittest import TestCase

import numpy as np
import pandas as pd


class TestFunctionApplication(TestCase):
    @staticmethod
    def add(series: pd.Series, term: Number) -> pd.Series:
        return series + term

    def test_custom_aggregation(self):
        df: pd.DataFrame = pd.DataFrame(np.arange(10).reshape((-1, 1)), columns=['A'])
        q_90: Callable[[pd.Series], Number] = partial(pd.Series.quantile, q=0.9, interpolation='lower')
        q_90.__name__ = '90%'
        result: pd.DataFrame = df.agg([q_90])

        self.assertEqual(result['A'][0], 8)

    def test_transform_df(self):
        df: pd.DataFrame = pd.DataFrame(np.ones((2, 2)), columns=['A', 'B'], dtype=int)

        add_1: Callable[[pd.Series], pd.Series] = partial(self.add, term=1)
        add_1.__name__ = 'add_1'
        add_2: Callable[[pd.Series], pd.Series] = partial(self.add, term=2)
        add_2.__name__ = 'add_2'
        add_3: Callable[[pd.Series], pd.Series] = partial(self.add, term=3)
        add_3.__name__ = 'add_3'

        # MultiIndex DataFrame is generated
        result: pd.DataFrame = df.transform({'A': add_1, 'B': [add_2, add_3]}).astype(int)
        target_result: pd.DataFrame = pd.DataFrame({
            ('A', 'add_1'): np.full(2, 2, dtype=int),
            ('B', 'add_2'): np.full(2, 3, dtype=int),
            ('B', 'add_3'): np.full(2, 4, dtype=int),
        })
        self.assertTrue(result.equals(target_result))

    def test_transform_series_lambdas(self):
        series: pd.Series = pd.Series(np.ones(2, dtype=int))
        result: pd.DataFrame = series.transform([lambda x: x + 1, lambda x: x + 2])

        self.assertEqual(result.shape, (2, 1))
        self.assertEqual(result.columns[0], '<lambda>')
        self.assertTrue((result.iloc[:, 0] == 3).all())

    def test_transform_series_named_functions(self):
        series: pd.Series = pd.Series(np.ones(2, dtype=int))

        add_1: Callable[[pd.Series], pd.Series] = partial(self.add, term=1)
        add_1.__name__ = 'add_1'
        add_2: Callable[[pd.Series], pd.Series] = partial(self.add, term=2)
        add_2.__name__ = 'add_2'

        result: pd.DataFrame = series.transform([add_1, add_2]).astype(int)
        target_result: pd.DataFrame = pd.DataFrame({'add_1': series + 1,
                                                    'add_2': series + 2})

        self.assertTrue(result.equals(target_result))
