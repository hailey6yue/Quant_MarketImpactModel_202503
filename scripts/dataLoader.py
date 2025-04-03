import pandas as pd
import numpy as np
import json
import os
import sys
from tqdm import tqdm
# Add parent directory to sys.path for custom imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Helper.MyDirectories import MyDirectories
from Utilities.TAQQuotesReader import TAQQuotesReader
from Utilities.TAQTradesReader import TAQTradesReader
from Utilities.VWAP import VWAP
from Utilities.TickTest import TickTest
from Helper.MidQuote import MidQuote
from Utilities.ReturnBuckets import ReturnBuckets

class DataLoader:
    def __init__(self, cache_file='stock_list.json', output_dir='../input', top_n=1500):
        """
        Initialize the feature extractor.

        Parameters:
        - cache_file: File to cache stock list
        - output_dir: Where to save output feature CSVs
        - top_n: Number of top liquid stocks to keep
        """
        self.trades_dir = MyDirectories.getTradesDir()
        self.quotes_dir = MyDirectories.getQuotesDir()
        self.cache_file = cache_file
        self.output_dir = output_dir
        self.top_n = top_n

        # Containers for features
        self.features = {
            'midquote_return': {},
            'total_daily_volume': {},
            'arrival_price': {},
            'imbalance': {},
            'VWAP_1': {},
            'VWAP_2': {},
            'terminal_price': {},
        }

        # Intraday time window boundaries
        self.startTS = 9.5 * 60 * 60 * 1000
        self.endTS_1 = 15.5 * 60 * 60 * 1000
        self.endTS_2 = 16 * 60 * 60 * 1000

    def get_top_liquid_stocks(self):
        """
        Determine the most liquid stocks based on average daily trading volume.
        Caches results in a JSON file to save time on future runs.
        """

        # First, check if we've already computed and cached the top stocks
        # If so, load them directly from the cache to avoid redundant processing
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                return json.load(f)

        # Dictionary to collect daily volume data for each stock
        # This will help us compute the average volume later
        volume_dict = {}

        # Get a sorted list of all available trading dates
        # Sorting ensures consistent processing order
        dates = sorted(os.listdir(self.trades_dir))

        # Loop through each date to aggregate trading volume
        for date in tqdm(dates, desc='Scanning Volumes'):
            data_path = os.path.join(self.trades_dir, date)

            # Skip entries that are not directories (e.g., hidden files or summaries)
            if not os.path.isdir(data_path):
                continue

            # Now loop through each file for this date to extract trade data
            for file in tqdm(os.listdir(data_path), desc=f'Volume on {date}', leave=False):

                # Only process trade files with the specific naming convention
                if file.endswith('_trades.binRT'):
                    # Extract the stock ticker from the filename
                    stock = file.replace('_trades.binRT', '')

                    # Build the full path to the trade data file
                    trade_file = os.path.join(data_path, file)

                    # Initialize the trade reader to access trade records
                    trades_reader = TAQTradesReader(trade_file)

                    # Sum the sizes of all trades for this stock on this date
                    daily_volume = sum(trades_reader.getSize(i) for i in range(trades_reader.getN()))

                    # Append the daily volume to our volume dictionary
                    # We collect a list of daily volumes for each stock
                    volume_dict.setdefault(stock, []).append(daily_volume)

        # Once all data is collected, compute the average daily volume per stock
        avg_daily_volume = {stock: np.mean(v) for stock, v in volume_dict.items()}

        # Sort stocks by their average volume in descending order
        # This helps us identify the most actively traded (liquid) stocks
        sorted_stocks = sorted(avg_daily_volume.items(), key=lambda x: x[1], reverse=True)

        # Select only the top N most liquid stocks for further processing
        top_stocks = [stock for stock, _ in sorted_stocks[:self.top_n]]

        # Cache the result to a JSON file to speed up future runs
        with open(self.cache_file, 'w') as f:
            json.dump(top_stocks, f)

        # Return the list of top stocks
        return top_stocks

    def process_all(self):
        """
        Loop through all available dates and top liquid stocks.
        For each stock-date pair, load the trade and quote files and extract features.
        """

        # First, retrieve the list of top N most liquid stocks
        # These are the only stocks we'll consider for feature extraction
        stock_list = self.get_top_liquid_stocks()

        # List and sort all dates available in the trade directory
        # Sorting ensures consistent processing order
        dates = sorted(os.listdir(self.trades_dir))

        # Loop through each date to process stock data for that day
        for date in tqdm(dates, desc='Processing Dates'):

            # Construct full paths for trade and quote directories for this date
            trade_dir = os.path.join(self.trades_dir, date)
            quote_dir = os.path.join(self.quotes_dir, date)

            # Skip this date if the trade directory doesn't exist
            # This helps handle cases where data might be incomplete
            if not os.path.isdir(trade_dir):
                continue

            # Now loop through each selected stock for the current date
            for stock in tqdm(stock_list, desc=f'Processing {date}', leave=False):

                # Build file paths for the trade and quote data for the given stock
                trade_file = os.path.join(trade_dir, f'{stock}_trades.binRT')
                quote_file = os.path.join(quote_dir, f'{stock}_quotes.binRQ')

                # Skip this stock if either trade or quote data is missing
                # Ensures we only process pairs with complete information
                if not os.path.exists(trade_file) or not os.path.exists(quote_file):
                    continue

                # Process the current stock on the current date
                # This will extract all relevant features and store them
                self.process_stock_date(stock, date, trade_file, quote_file)

    def process_stock_date(self, stock, date, trade_file, quote_file):
        """
        Extracts various features for a single stock on a given date.
        These include returns, volume, VWAP, arrival/terminal prices, and imbalance.
        """

        # Load TAQ trade and quote data for the specified stock and date
        trades_reader = TAQTradesReader(trade_file)
        quotes_reader = TAQQuotesReader(quote_file)

        # Generate midquote series from the quotes
        # These midquotes will later be used to compute returns and price-based features
        midquote_data = MidQuote(quotes_reader)

        # Compute return buckets between start and end of day (195 intervals)
        # This gives us a time-resolved view of price changes
        midquote_return = ReturnBuckets(midquote_data, self.startTS, self.endTS_2, 195)

        # Extract non-null returns from the bucketed return series
        # These represent intraday price changes and will be used as features
        midquote_return_list = [
            midquote_return.getReturn(i)
            for i in range(midquote_return.getN())
            if midquote_return.getReturn(i) is not None
        ]

        # Save the list of returns if available; else store NaN to indicate missing data
        self.features['midquote_return'].setdefault(stock, {})[date] = (
            midquote_return_list if midquote_return_list else np.nan
        )

        # Sum the sizes of all trades throughout the day to get total trading volume
        # This acts as a proxy for liquidity
        volume = sum(trades_reader.getSize(i) for i in range(trades_reader.getN()))

        # Save volume only if it's non-zero, to avoid noise from empty or faulty files
        if volume:
            self.features['total_daily_volume'].setdefault(stock, {})[date] = volume

        # Get the first 5 valid midquote prices to estimate the arrival price
        # This captures the price level near market open
        arrival_prices = [
            midquote_data.getPrice(i)
            for i in range(5)
            if midquote_data.getPrice(i) is not None
        ]

        # If we have enough data points, save the average as the arrival price
        if len(arrival_prices) == 5:
            self.features['arrival_price'].setdefault(stock, {})[date] = np.mean(arrival_prices)

        # Initialize imbalance counter
        # This will measure the net buy/sell pressure throughout the day
        imbalance = 0

        # Use the Tick Test to classify trades as buyer- or seller-initiated
        # This helps in computing directional volume imbalance
        trades = TickTest().classifyAll(trades_reader, self.startTS, self.endTS_1)
        for i, (_, _, side) in enumerate(trades):
            size = trades_reader.getSize(i)
            imbalance += side * size  # +1 for buy, -1 for sell

        # Save imbalance only if it's non-zero
        if imbalance:
            self.features['imbalance'].setdefault(stock, {})[date] = imbalance

        # Calculate VWAP from start of day to 3:30pm (endTS_1)
        # This provides a volume-weighted average price over most of the trading day
        VWAP_1 = VWAP(trades_reader, self.startTS, self.endTS_1).getVWAP()
        if VWAP_1:
            self.features['VWAP_1'].setdefault(stock, {})[date] = VWAP_1

        # Calculate VWAP from start of day to 4:00pm (endTS_2) to include closing period
        VWAP_2 = VWAP(trades_reader, self.startTS, self.endTS_2).getVWAP()
        if VWAP_2:
            self.features['VWAP_2'].setdefault(stock, {})[date] = VWAP_2

        # Get the last 5 valid midquote prices before close
        # These are used to compute the terminal price (near market close)
        N = midquote_data.getN()
        terminal_prices = [
            midquote_data.getPrice(i)
            for i in range(N - 5, N)
            if midquote_data.getPrice(i) is not None
        ]

        # Store the average of the final prices as terminal price
        if len(terminal_prices) == 5:
            self.features['terminal_price'].setdefault(stock, {})[date] = np.mean(terminal_prices)
        ## test_hug3874

    def save_features(self):
        """
        Save all extracted features as individual CSVs.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        for name, data in self.features.items():
            df = pd.DataFrame(data).transpose()
            df.to_csv(os.path.join(self.output_dir, f'{name}.csv'))

    def run(self):
        """
        Main method to extract features and save them.
        """
        print("Extracting features from TAQ data...")
        self.process_all()
        print("Saving results...")
        self.save_features()
        print("Feature extraction completed.")


# Entry point
if __name__ == "__main__":
    extractor = DataLoader()
    extractor.run()