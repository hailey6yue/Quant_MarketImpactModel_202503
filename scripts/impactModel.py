import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white
import os
import sys
# Add parent directory to sys.path (optional, useful if you import custom modules from outside)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class MarketImpactModel:
    def __init__(self, data_path='../input.csv', seed=42, trim=True):
        """
        Initialize the MarketImpactModel.

        Parameters:
        - data_path (str): Path to the input CSV containing TAQ-derived features.
        - seed (int): Random seed for reproducibility in bootstrapping.
        - trim (bool): Whether to clip extreme values of Z and sigma to reduce outlier influence.
        """
        self.df = pd.read_csv(data_path)
        self.seed = seed
        self.trim = trim
        self.eta_hat = None
        self.beta_hat = None
        self.t_eta = None
        self.t_beta = None
        self.etas = []
        self.betas = []
        self._prepare_data()

    def _prepare_data(self):
        """
        Compute derived variables needed for model estimation:
        - X (signed dollar imbalance)
        - V (dollar volume)
        - Z (normalized imbalance)
        - h (temporary impact: VWAP deviation net of permanent impact)
        Optionally trims extreme values in Z and sigma to prevent large residual variance caused by outliers.
        """
        df = self.df

        # Construct intermediate variables
        df['X'] = df['daily_imbalance_value']  # Signed dollar imbalance
        df['V'] = df['daily_value_added']  # Daily traded dollar value
        df['sigma'] = df['volatility']  # Volatility estimate (e.g., from return buckets)
        df['T'] = 6 / 6.5  # Execution window length (assumed 6h in a 6.5h day)
        df['Z'] = df['X'] / (df['T'] * df['V'])  # Normalized signed imbalance
        df['h'] = (df['VWAP_1'] - df['arrival_price']) - (df['terminal_price'] - df['arrival_price']) / 2
        # h = temporary impact = VWAP deviation net of estimated permanent impact

        if self.trim:
            # Clip extreme values of Z and sigma to 10th and 90th percentiles to reduce noise from outliers
            df['Z'] = df['Z'].clip(lower=df['Z'].quantile(0.10), upper=df['Z'].quantile(0.90))
            df['sigma'] = df['sigma'].clip(lower=df['sigma'].quantile(0.10), upper=df['sigma'].quantile(0.90))

        # Drop any rows with missing values after computation
        self.df = df.dropna()

    @staticmethod
    def impact_function(x, eta, beta):
        """
        The Almgren-Chriss nonlinear market impact function.

        Parameters:
        - x (tuple): A tuple of arrays (Z, sigma)
        - eta (float): Volatility scaling coefficient
        - beta (float): Power-law nonlinearity coefficient

        Returns:
        - Estimated temporary impact for each observation
        """
        Z, sigma = x
        return eta * sigma * (np.abs(Z) ** beta) * np.sign(Z)
        ## test_sfh1734

    def fit(self, p0=[0.142, 0.6], maxfev=10000):
        """
        Fit the nonlinear market impact model using curve_fit on the entire dataset.

        Parameters:
        - p0 (list): Initial guess for [eta, beta]
        - maxfev (int): Max function evaluations for curve fitting
        """
        X = self.df['Z'].values
        sigma = self.df['sigma'].values
        y = self.df['h'].values

        # Estimate parameters eta and beta via nonlinear least squares
        params, _ = curve_fit(self.impact_function, (X, sigma), y, p0=p0, maxfev=maxfev)
        self.eta_hat, self.beta_hat = params
        ## test_jrs1345

    def _fit_regime(self,df_subset, p0=[0.142, 0.6], maxfev=10000):
        """
        Internal helper to fit the model on a given subset of the dataset.

        Parameters:
        - df_subset (pd.DataFrame): Filtered subset of self.df
        - p0 (list): Initial parameter guess [eta, beta]
        - maxfev (int): Max number of function evaluations

        Returns:
        - Tuple (eta, beta): Fitted model parameters
        """
        X = df_subset['Z'].values
        sigma = df_subset['sigma'].values
        y = df_subset['h'].values
        params, _ = curve_fit(self.impact_function, (X, sigma), y, p0=p0, maxfev=maxfev)
        return params
        ## test_owi2853

    def fit_by_volatility(self, threshold=0.038):
        """
        Fit the model separately on normal and high-volatility days.

        Parameters:
        - threshold (float): Cutoff for volatility to distinguish regimes.

        Returns:
        - Dictionary of estimated parameters {eta, beta} for both regimes.
        """
        df = self.df.copy()
        df_normal = df[df['sigma'] < threshold]
        df_volatile = df[df['sigma'] >= threshold]

        # Fit both regimes
        eta_normal, beta_normal = self._fit_regime(df_normal)
        eta_vol, beta_vol = self._fit_regime(df_volatile)

        # Print and return result
        print(f"Normal days:   eta = {eta_normal:.4f}, beta = {beta_normal:.4f}")
        print(f"Volatile days: eta = {eta_vol:.4f}, beta = {beta_vol:.4f}")

        return {
            'eta_normal': eta_normal, 'beta_normal': beta_normal,
            'eta_volatile': eta_vol, 'beta_volatile': beta_vol
        }
        ## test_kwg8327

    def fit_by_activity(self, threshold=None):
        """
        Fit the model separately on active and inactive stocks based on daily traded value.

        Parameters:
        - threshold (float or None): Value cutoff for separating active vs. inactive stocks.
                                     If None, defaults to the median of 'daily_value_added'.

        Returns:
        - Dictionary of estimated parameters {eta, beta} for both regimes.
        """
        df = self.df.copy()

        # Use median as default threshold if not specified
        if threshold is None:
            threshold = df['daily_value_added'].median()

        # Split into active and inactive regimes
        df_inactive = df[df['daily_value_added'] < threshold]
        df_active = df[df['daily_value_added'] >= threshold]

        # Fit the model separately for inactive and active subsets
        eta_inactive, beta_inactive = self._fit_regime(df_inactive)
        eta_active, beta_active = self._fit_regime(df_active)

        # Print and return result
        print(f"Inactive stocks: eta = {eta_inactive:.4f}, beta = {beta_inactive:.4f}")
        print(f"Active stocks:   eta = {eta_active:.4f}, beta = {beta_active:.4f}")

        return {
            'eta_inactive': eta_inactive, 'beta_inactive': beta_inactive,
            'eta_active': eta_active, 'beta_active': beta_active
        }
        ## test_iwg8257

    def bootstrap(self, n_boot=500):
        """
        Estimate standard errors and t-statistics for eta and beta using paired bootstrap.

        Parameters:
        - n_boot (int): Number of bootstrap iterations.
        """
        # Extract input features (Z, sigma) and response (h) from preprocessed dataframe
        X = self.df['Z'].values  # Normalized imbalance
        sigma = self.df['sigma'].values  # Volatility
        y = self.df['h'].values  # Temporary market impact

        # Initialize random number generator for reproducibility
        rng = np.random.default_rng(self.seed)
        n = len(X)  # Number of observations

        # Containers to store bootstrap estimates of eta and beta
        etas, betas = [], []

        # Perform bootstrap resampling n_boot times
        for _ in range(n_boot):
            # Randomly sample n indices with replacement
            idx = rng.integers(0, n, size=n)

            try: ## test_wfb3875
                # Fit the model on the resampled dataset
                p, _ = curve_fit(
                    self.impact_function,  # Nonlinear impact model
                    (X[idx], sigma[idx]),  # Paired input features
                    y[idx],  # Corresponding responses
                    p0=[self.eta_hat, self.beta_hat],  # Use fitted params as initial guess
                    maxfev=10000  # Max evaluations for convergence
                )
                # Store the fitted parameters
                etas.append(p[0])
                betas.append(p[1])
            except: ## test_ghy2764
                # Skip this sample if fitting fails (e.g., due to poor conditioning)
                continue

        # Convert collected bootstrap estimates to numpy arrays
        self.etas = np.array(etas)
        self.betas = np.array(betas)

        # Compute bootstrap standard errors (sample standard deviation)
        eta_se = self.etas.std(ddof=1)
        beta_se = self.betas.std(ddof=1)

        # Compute t-statistics: parameter estimate divided by its bootstrap SE
        self.t_eta = self.eta_hat / eta_se
        self.t_beta = self.beta_hat / beta_se

    def save_results(self, output_path='params_part1.txt'):
        """
        Save the estimated parameters and t-statistics to a text file.

        Parameters:
        - output_path (str): Path to output text file.
        """
        with open(output_path, "w") as f:
            f.write(f"eta = {self.eta_hat:.6f}\n")
            f.write(f"t-eta = {self.t_eta:.6f}\n")
            f.write(f"beta = {self.beta_hat:.6f}\n")
            f.write(f"t-beta = {self.t_beta:.6f}\n")

    def run_whites_test(self, df, x_cols, y_col='h', print_result=True):
        """
        Perform White's test for heteroskedasticity.

        Returns:
        - dict with test statistic, p-values, F-statistic, and F p-value.
        """
        # Prepare design matrix with intercept
        X = sm.add_constant(np.column_stack([df[col] for col in x_cols]))
        y = df[y_col]

        # Fit the OLS model
        model = sm.OLS(y, X).fit()

        # Perform White's test on residuals
        white_test = het_white(model.resid, model.model.exog)

        # Unpack results
        test_stat, p_val, f_stat, f_p_val = white_test

        # Optional printout
        if print_result:
            print(f"Test Statistic:       {test_stat:.4f}")
            print(f"Test Statistic p-val: {p_val:.4f}")
            print(f"F-Statistic:          {f_stat:.4f}")
            print(f"F-Test p-val:         {f_p_val:.4f}")

        return {
            "Test Statistic": test_stat,
            "p-value": p_val,
            "F-Statistic": f_stat,
            "F p-value": f_p_val
        }
        ## test_isu2725

# -------------------------------
# Example Usage
# -------------------------------
if __name__ == "__main__":
    # Initialize model with optional trimming of extreme values
    model = MarketImpactModel(data_path='../input.csv', trim=True)

    # Fit full dataset using nonlinear least squares
    model.fit()

    model.fit_by_activity()

    # Run paired bootstrap to estimate standard errors and t-values
    model.bootstrap(n_boot=500)

    # Save results to file in required format
    model.save_results()