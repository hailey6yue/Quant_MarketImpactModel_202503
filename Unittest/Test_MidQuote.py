import unittest
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Helper.MyDirectories import MyDirectories
from Helper.MidQuote import MidQuote
from Utilities.TAQQuotesReader import TAQQuotesReader

class Test_MidQuote(unittest.TestCase):

    def test_duh7246(self):
        reader = TAQQuotesReader(MyDirectories.getQuotesDir() + '/20070920/IBM_quotes.binRQ')
        data = MidQuote(reader)

        self.assertEqual(data.getN(),70166)
        self.assertEqual(data.getTimestamp(0), 34210000)
        self.assertEqual(data.getPrice(0), 116.19999694824219)

if __name__ == "__main__":
    unittest.main()
