import pandas as pd
import ast
import sys
import os

# Append parent directory to sys.path
# This allows us to import modules like GetVolatility that live in other folders
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import custom volatility computation class
from Helper.GetVolatility import GetVolatility

class DataPreprocessor:
    def __init__(self, input_dir='../input', output_path='../input.csv'):
        """
        Initialize the DataPreprocessor with input and output paths.

        Parameters:
        - input_dir: Folder where raw input CSVs are stored
        - output_path: Where the cleaned, merged output will be written
        """
        self.input_dir = input_dir        # Store the directory path to read raw data
        self.output_path = output_path    # Path to save the final merged output file
        self.dfs = {}                     # Dictionary to hold each input DataFrame
        self.df_input = None              # Will store the final long-format dataset

    def load_csv(self, filename):
        """
        Load a CSV file, drop rows with NaNs, and set the first column as index.

        This step ensures all input files are in a clean, consistent format before merging.
        """
        # Construct full file path and read CSV
        df = pd.read_csv(os.path.join(self.input_dir, filename))
        df.dropna(inplace=True)                  # Drop any missing values to avoid merge issues
        df.set_index(df.columns[0], inplace=True)  # Set the first column (usually stock name) as index
        return df

    def load_all_data(self):
        """
        Load all required input CSVs and store them in self.dfs.

        This includes price, volume, imbalance, and other market features.
        """
        # Call load_csv on each required input and store result in dictionary
        self.dfs['arrival_price'] = self.load_csv('arrival_price.csv')
        self.dfs['imbalance'] = self.load_csv('imbalance.csv')
        self.dfs['midquote_return'] = self.load_csv('midquote_return.csv')
        self.dfs['terminal_price'] = self.load_csv('terminal_price.csv')
        self.dfs['total_daily_volume'] = self.load_csv('total_daily_volume.csv')
        self.dfs['VWAP_1'] = self.load_csv('VWAP_1.csv')
        self.dfs['VWAP_2'] = self.load_csv('VWAP_2.csv')

    def compute_derived_features(self):
        """
        Compute additional columns required for modeling.

        These include daily traded value, dollar imbalance, and volatility.
        """
        df = self.dfs  # Shorthand for dictionary

        # Compute traded value = VWAP × volume; a proxy for market activity
        df['daily_value_added'] = df['VWAP_2'] * df['total_daily_volume']

        # Compute dollar imbalance = VWAP × signed volume imbalance
        df['daily_imbalance_value'] = df['VWAP_2'] * df['imbalance']
        ## test_jug3478

    def compute_volatility(self, isSeries=False):
        """
        Parse midquote return strings and compute volatility.
        """
        df = self.dfs

        # Convert stringified return lists (e.g., "[0.1, -0.2, ...]") to real Python lists
        df['midquote_return'] = df['midquote_return'].applymap(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

        # Compute volatility using custom GetVolatility class
        # Uses rolling window over 10 days and 195 intervals per day
        df['volatility'] = GetVolatility(
            df['midquote_return'],
            window=10,
            n_intervals=195
        ).compute()

    def melt_to_long_format(self, df, name):
        """
        Convert wide-format DataFrame (stock × date) to long-format.

        Parameters:
        - df: wide DataFrame to convert
        - name: the name of the resulting value column

        Returns:
        - A DataFrame with columns: stock, date, name
        """
        df_reset = df.reset_index()                            # Convert index back to column
        df_reset.rename(columns={"Unnamed: 0": "stock"}, inplace=True)  # Rename index to 'stock'
        return pd.melt(                                        # Convert to long format
            df_reset,
            id_vars='stock',                                   # Keep 'stock' as identifier
            var_name='date',                                   # Columns become 'date'
            value_name=name                                    # Values go into a new column named after feature
        )
    ## test_sig8234

    def merge_all_inputs(self):
        """
        Merge all features into one long-format table.

        This table will have one row per stock-date-feature triple.
        """
        # These are the features we want to merge into the final dataset
        input_keys = [
            'arrival_price', 'terminal_price', 'VWAP_1',
            'daily_value_added', 'daily_imbalance_value', 'volatility'
        ]

        df_merged = None

        # Loop through all selected input features
        for key in input_keys:
            # Convert each to long format
            long_df = self.melt_to_long_format(self.dfs[key], key)
            if df_merged is None:
                # If this is the first feature, initialize the merge
                df_merged = long_df
            else:
                # Otherwise, merge on stock and date
                df_merged = df_merged.merge(long_df, on=['stock', 'date'], how='outer')

        # Save final merged DataFrame
        self.df_input = df_merged

    def save_to_csv(self):
        """
        Save the merged dataset to a CSV file.

        Used for downstream model fitting or analysis.
        """
        self.df_input.to_csv(self.output_path, index=False)

    def run(self):
        """
        Full pipeline to load, process, merge, and save all features.
        """
        print("Loading raw data...")               # Step 1: Read raw CSVs
        self.load_all_data()

        print("Computing derived features...")     # Step 2: Compute value-based features
        self.compute_derived_features()

        print("Computing volatilites...")  # Step 3: Compute extra fields like volatility
        self.compute_volatility()

        print("Merging into long-format table...") # Step 4: Reshape and combine all data
        self.merge_all_inputs()

        print(f"Saving processed dataset to {self.output_path}")  # Step 4: Write final file
        self.save_to_csv()

        print("Done. Final dataset preview:")      # Final check
        print(self.df_input.head())

# -------------------------
# Run the Preprocessing Pipeline
# -------------------------
if __name__ == "__main__":
    # Instantiate and run the full processing pipeline
    processor = DataPreprocessor()
    processor.run()