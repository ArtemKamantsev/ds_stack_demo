from collections import namedtuple
from unittest import TestCase
from scipy.sparse import csr_matrix, coo_matrix

import numpy as np
import pandas as pd


class TestDataFrame(TestCase):
    def test_discard_data_on_creation(self) -> None:
        df: pd.DataFrame = pd.DataFrame({'A': [1, 2], 'B': [3, 4]}, columns=['A'])

        # column 'B' has been discarded
        self.assertEqual(df.shape[1], 1)
        self.assertTrue('A' in df)
        self.assertFalse('B' in df)

    def test_creation_from_series_with_different_index(self) -> None:
        series1: pd.Series = pd.Series([0, 1], index=[0, 1])
        series2: pd.Series = pd.Series([1, 2], index=[1, 2])
        df: pd.DataFrame = pd.DataFrame({'A': series1, 'B': series2})

        df_target: pd.DataFrame = pd.DataFrame([
            [0, np.nan],
            [1, 1],
            [np.nan, 2]
        ], columns=['A', 'B'])
        self.assertTrue(df.equals(df_target))

    def test_creation_from_named_tuples_of_different_size(self) -> None:
        Short = namedtuple('Short', 'a')
        Long = namedtuple('Long', 'a b')

        short: Short = Short(0)
        long: Long = Long(0, 1)

        df: pd.DataFrame = pd.DataFrame([long, short])
        self.assertEqual(df.isna().sum().sum(), 1)
        self.assertTrue(pd.isna(df['b'][1]))

        with self.assertRaises(ValueError):
            # If any item is longer than the first namedtuple, a ValueError is raised.
            df = pd.DataFrame([short, long])

    def test_column_assignment(self) -> None:
        data: pd.DataFrame = pd.DataFrame({
            'k': [1, 2],
            'b': [1, 2],
            'x': [1, 2],
        })

        # Starting with Python 3.6 the order of **kwargs is preserved.
        # This allows for dependent assignment, where an expression later in **kwargs
        # can refer to a column created earlier in the same assign().
        data_enhanced: pd.DataFrame = data.assign(**{'kx': lambda row: row['k'] * row['x'],
                                                     'kx+b': lambda row: row['kx'] + row['b']})
        self.assertTrue((data_enhanced[['kx', 'kx+b']] == np.array([[1, 2], [4, 6]])).all().all())

    def test_series_broadcasting(self) -> None:
        df: pd.DataFrame = pd.DataFrame(np.arange(20).reshape((10, 2)), columns=["A", "B"], dtype=int)

        result: pd.DataFrame = df - df.iloc[0]
        self.assertTrue((result.iloc[0] == 0).all())

    def test_df_equality(self) -> None:
        df1: pd.DataFrame = pd.DataFrame({"A": [1, 2, 3]})
        df2: pd.DataFrame = pd.DataFrame({"A": [3, 2, 1]}, index=[2, 1, 0])

        self.assertFalse(df1.equals(df2))
        self.assertTrue(df1.equals(df2.sort_index()))  # index order is essential

    def test_combine_first(self) -> None:
        df1: pd.DataFrame = pd.DataFrame([[0, np.nan],
                                          [np.nan, 1]], dtype='Int64')
        df2: pd.DataFrame = pd.DataFrame([[1, np.nan],
                                          [2, np.nan]], dtype='Int64')

        result: pd.DataFrame = df1.combine_first(df2)
        target_result: pd.DataFrame = pd.DataFrame([[0, np.nan],
                                                    [2, 1]], dtype='Int64')

        self.assertTrue(result.equals(target_result))

    def test_combine(self) -> None:
        df1: pd.DataFrame = pd.DataFrame([[0, np.nan],
                                          [np.nan, 1]], dtype='Int64')
        df2: pd.DataFrame = pd.DataFrame([[1, np.nan],
                                          [2, np.nan]], dtype='Int64')

        def combiner(primary: pd.Series, secondary: pd.Series) -> np.ndarray:
            return np.where(pd.isna(primary), secondary, primary)

        result = df1.combine(df2, combiner).astype('Int64')
        target_result: pd.DataFrame = pd.DataFrame([[0, np.nan], [2, 1]], dtype='Int64')

        self.assertTrue(result.equals(target_result))

    def test_default_iteration(self):
        data: np.ndarray = np.zeros((2,))
        df: pd.DataFrame = pd.DataFrame({'A': data, 'B': data})

        self.assertTrue((np.asarray(list(df)) == ['A', 'B']).all())

    def test_interaction_with_scipy_sparse(self):
        dense: np.ndarray = np.arange(9).reshape((3, 3))
        dense[dense < 4] = 0
        sparse: csr_matrix = csr_matrix(dense)

        result: pd.DataFrame = pd.DataFrame.sparse.from_spmatrix(sparse)
        target_result: pd.DataFrame = pd.DataFrame([
            [0, 0, 0],
            [0, 4, 5],
            [6, 7, 8],
        ], dtype="Sparse[int]")

        self.assertTrue(result.equals(target_result))
        self.assertIsInstance(result.sparse.to_coo(), coo_matrix)
