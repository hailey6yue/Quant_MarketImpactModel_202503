import unittest
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))
from dataLoader import DataLoader
from Helper.MyDirectories import MyDirectories

class TestProcessStockDate(unittest.TestCase):
    def test_hug3874(self):
        # Setup file paths
        stock = 'IBM'
        date = '20070919'
        trade_file = os.path.join(MyDirectories.getTradesDir(), date, f'{stock}_trades.binRT')
        quote_file = os.path.join(MyDirectories.getQuotesDir(), date, f'{stock}_quotes.binRQ')

        # Ensure files exist
        self.assertTrue(os.path.exists(trade_file))
        self.assertTrue(os.path.exists(quote_file))

        # Create extractor and manually set time boundaries
        extractor = DataLoader()
        extractor.startTS = 9.5 * 60 * 60 * 1000
        extractor.endTS_1 = 15.5 * 60 * 60 * 1000
        extractor.endTS_2 = 16 * 60 * 60 * 1000

        # Run the feature extraction on one stock-date pair
        extractor.process_stock_date(stock, date, trade_file, quote_file)

        # Extracted features
        feats = extractor.features

        # Check values
        self.assertAlmostEqual(feats['arrival_price'][stock][date], 116.70, places=2)
        self.assertAlmostEqual(feats['terminal_price'][stock][date], 116.67, places=2)
        self.assertAlmostEqual(feats['VWAP_1'][stock][date], 116.45, places=2)
        self.assertAlmostEqual(feats['imbalance'][stock][date], -388366, places=1)
        self.assertAlmostEqual(feats['total_daily_volume'][stock][date], 8960834, places=1)

        # Ensure return list is non-empty and all values are floats
        mid_returns = feats['midquote_return'][stock][date]
        self.assertIsInstance(mid_returns, list)
        self.assertGreater(len(mid_returns), 0)
        self.assertTrue(all(isinstance(r, float) for r in mid_returns))

if __name__ == '__main__':
    unittest.main()