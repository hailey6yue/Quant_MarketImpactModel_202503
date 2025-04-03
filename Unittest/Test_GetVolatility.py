import pandas as pd
import numpy as np
import unittest
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Helper.GetVolatility import GetVolatility

class TestGetVolatility(unittest.TestCase):
    def test_shr8457(self):
        # Fake returns: 2 stocks × 3 days, 3 returns per day
        data = {
            '20240101': [[0.01, 0.01, 0.01], [0.02, 0.02, 0.02]],
            '20240102': [[0.02, 0.02, 0.02], [0.03, 0.03, 0.03]],
            '20240103': [[0.03, 0.03, 0.03], [0.01, 0.01, 0.01]],
        }
        df = pd.DataFrame(data, index=['AAPL', 'MSFT'])

        # window=2, n_intervals=3
        gv = GetVolatility(df, window=2, n_intervals=3)
        result = gv.compute()

        # Manually compute expected vol for AAPL on 20240103
        # Last two days: [0.02×3 + 0.03×3] = [0.02, 0.02, 0.02, 0.03, 0.03, 0.03]
        aapl_returns = np.array([0.02, 0.02, 0.02, 0.03, 0.03, 0.03])
        sigma = np.std(aapl_returns, ddof=1)
        expected_vol = sigma * np.sqrt(3)

        # First column should be NaN
        self.assertTrue(np.isnan(result.loc['AAPL', '20240101']))
        # Check final volatility value (rounded to 5 decimals)
        self.assertAlmostEqual(result.loc['AAPL', '20240103'], expected_vol, places=5)

if __name__ == "__main__":
    unittest.main()