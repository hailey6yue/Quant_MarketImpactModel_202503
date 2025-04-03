import pandas as pd
import numpy as np

class GetVolatility:
    def __init__(self, df, window=10, n_intervals=195):
        """
        Initialize the volatility estimator.

        Parameters:
        ----------
        df : pd.DataFrame
            Wide-format input where:
              - Rows = stocks
              - Columns = dates
              - Each cell = list of intraday returns (e.g., 2-minute returns)

        window : int, default=10
            Size of the rolling window in days (used to compute historical volatility).

        n_intervals : int, default=195
            Number of 2-minute intervals in a trading day (used to annualize volatility).
            For example: 6.5 hours * 60 / 2 = 195
        """
        self.df = df
        self.window = window
        self.n_intervals = n_intervals

    def _rolling_volatility(self, series):
        """
        Internal method to compute rolling volatility for a single stock (i.e., one row).

        Parameters:
        ----------
        series : pd.Series
            A row from the input DataFrame.
            Each element in the series is a list of 2-minute returns for one day.

        Returns:
        -------
        pd.Series
            Rolling volatility for that stock across time (dates).
            First (window - 1) values will be NaN due to lack of lookback window.
        """
        vol_list = []

        # Iterate over each day
        for i in range(len(series)):
            if i < self.window - 1:
                # Not enough data to compute volatility — pad with NaN
                vol_list.append(np.nan)
            else:
                # Gather `window` days of return data
                window_data = series.iloc[i - self.window + 1 : i + 1]

                # Flatten all return lists in the window into a single array
                returns = np.concatenate(window_data.to_list())

                # Compute standard deviation of returns
                sigma = np.std(returns, ddof=1)

                # Scale by sqrt(n_intervals) to get daily volatility estimate
                vol = sigma * np.sqrt(self.n_intervals)

                # Store result
                vol_list.append(vol)

        # Return a Series of the same index as the input series
        return pd.Series(vol_list, index=series.index)

    def compute(self):
        """
        Apply rolling volatility computation to all stocks (i.e., all rows).

        Returns:
        -------
        pd.DataFrame
            Same shape as input DataFrame (stocks × dates),
            but each cell now contains the rolling volatility.
        """
        return self.df.apply(self._rolling_volatility, axis=1)