import pandas as pd
import unittest
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))
from dataPreprocessor import DataPreprocessor

class TestDataPreprocessor(unittest.TestCase):
    def test_jug3478(self):
        # Fake input data with normal layout
        data = {
            'VWAP_2': [100.0, 200.0],
            'total_daily_volume': [1000, 2000],
            'imbalance': [100, -150],
        }
        df = pd.DataFrame(data, index=['AAPL', 'MSFT'])  # Stocks as rows

        dp = DataPreprocessor()
        dp.dfs = df

        # Call the method (it modifies dp.dfs in place)
        dp.compute_derived_features()
        result_df = dp.dfs

        # Assertions
        self.assertEqual(result_df.loc['AAPL', 'daily_value_added'], 100.0 * 1000)
        self.assertEqual(result_df.loc['MSFT', 'daily_value_added'], 200.0 * 2000)
        self.assertEqual(result_df.loc['AAPL', 'daily_imbalance_value'], 100.0 * 100)
        self.assertEqual(result_df.loc['MSFT', 'daily_imbalance_value'], 200.0 * -150)

    def test_sig8234(self):
        # Fake wide-format input
        data = {
            '20240101': [1, 2],
            '20240102': [3, 4]
        }
        df = pd.DataFrame(data, index=['AAPL', 'MSFT'])
        df.index.name = 'Unnamed: 0'  # Simulate original index name

        dp = DataPreprocessor()
        melted = dp.melt_to_long_format(df, name='volume')

        # Expected long-format DataFrame
        expected = pd.DataFrame({
            'stock': ['AAPL', 'MSFT', 'AAPL', 'MSFT'],
            'date': ['20240101', '20240101', '20240102', '20240102'],
            'volume': [1, 2, 3, 4]
        })

        # Sort and reset for safe comparison
        melted_sorted = melted.sort_values(['stock', 'date']).reset_index(drop=True)
        expected_sorted = expected.sort_values(['stock', 'date']).reset_index(drop=True)

        pd.testing.assert_frame_equal(melted_sorted, expected_sorted)

if __name__ == "__main__":
    unittest.main()