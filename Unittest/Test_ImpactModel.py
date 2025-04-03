import unittest
import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))
from impactModel import MarketImpactModel

class Test_MarketImpactModel(unittest.TestCase):
    def test_sfh1734(self):
        # Simulated test input
        Z = np.array([1.0, -2.0, 0.0])  # trading volume
        sigma = np.array([0.2, 0.5, 0.1])  # volatility
        eta = 0.1  # scale
        beta = 0.5  # power law

        # Expected output manually calculated:
        # result = eta * sigma * (abs(Z)^beta) * sign(Z)
        # => [0.1 * 0.2 * (1.0)^0.5 * 1,
        #     0.1 * 0.5 * (2.0)^0.5 * -1,
        #     0.1 * 0.1 * (0.0)^0.5 * 0] = [0.02, -0.07071068, 0.0]
        expected = np.array([0.02, -0.07071068, 0.0])

        model = MarketImpactModel()
        result = model.impact_function((Z, sigma), eta, beta)

        # Check that each value is almost equal (floating-point safe)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_jrs1345(self):
        # Simulated test data
        Z = np.array([1.0, 2.0, 3.0])
        sigma = np.array([0.5, 0.5, 0.5])
        eta_true = 0.1
        beta_true = 0.5
        h = eta_true * sigma * (np.abs(Z) ** beta_true) * np.sign(Z)

        # Create dummy DataFrame and assign to model
        model = MarketImpactModel()
        model.df = pd.DataFrame({'Z': Z, 'sigma': sigma, 'h': h})

        # Call the fitting function with default starting params
        model.fit()

        # Check that the fitted parameters are close to the true values
        self.assertAlmostEqual(model.eta_hat, eta_true, places=2)
        self.assertAlmostEqual(model.beta_hat, beta_true, places=2)

    def test_owi2853(self):
        # Simulated subset of data for regime
        Z = np.array([1.0, 2.0, 3.0])
        sigma = np.array([1.0, 1.0, 1.0])
        eta_true = 0.2
        beta_true = 0.6
        h = eta_true * sigma * (np.abs(Z) ** beta_true) * np.sign(Z)

        # Create DataFrame subset as expected by _fit_regime
        df_subset = pd.DataFrame({
            'Z': Z,
            'sigma': sigma,
            'h': h
        })

        # Fit model using the internal _fit_regime method
        model = MarketImpactModel()
        eta_hat, beta_hat = model._fit_regime(df_subset)

        # Assert that fitted values are close to the true ones
        self.assertAlmostEqual(eta_hat, eta_true, places=2)
        self.assertAlmostEqual(beta_hat, beta_true, places=2)

    def test_kwg8327(self):
        # Simulate data covering both volatility regimes
        Z = np.array([1.0, 2.0, 1.0, 2.0])  # same trading size for both groups
        sigma = np.array([0.02, 0.025, 0.05, 0.06])  # two below threshold, two above
        eta_normal = 0.1
        beta_normal = 0.5
        eta_volatile = 0.3
        beta_volatile = 0.7

        # Generate h based on regime-specific parameters
        h = np.array([
            eta_normal * sigma[0] * (np.abs(Z[0]) ** beta_normal),
            eta_normal * sigma[1] * (np.abs(Z[1]) ** beta_normal),
            eta_volatile * sigma[2] * (np.abs(Z[2]) ** beta_volatile),
            eta_volatile * sigma[3] * (np.abs(Z[3]) ** beta_volatile)
        ])

        # Create full DataFrame
        df = pd.DataFrame({'Z': Z, 'sigma': sigma, 'h': h})

        # Fit using fit_by_volatility
        model = MarketImpactModel()
        model.df = df
        results = model.fit_by_volatility(threshold=0.038)

        # Assertions for normal regime
        self.assertAlmostEqual(results['eta_normal'], eta_normal, places=2)
        self.assertAlmostEqual(results['beta_normal'], beta_normal, places=2)

        # Assertions for volatile regime
        self.assertAlmostEqual(results['eta_volatile'], eta_volatile, places=2)
        self.assertAlmostEqual(results['beta_volatile'], beta_volatile, places=2)

    def test_iwg8257(self):
        # Simulated trading data
        Z = np.array([1.0, 2.0, 1.0, 2.0])
        sigma = np.array([0.2, 0.2, 0.2, 0.2])
        daily_value = np.array([100, 120, 300, 400])  # median = 210

        # Split:
        # inactive: [100, 120]
        # active:   [300, 400]

        # Ground truth parameters for each regime
        eta_inactive = 0.1
        beta_inactive = 0.5
        eta_active = 0.3
        beta_active = 0.7

        # Compute h using true parameters for each regime
        h = np.array([
            eta_inactive * sigma[0] * (np.abs(Z[0]) ** beta_inactive),
            eta_inactive * sigma[1] * (np.abs(Z[1]) ** beta_inactive),
            eta_active * sigma[2] * (np.abs(Z[2]) ** beta_active),
            eta_active * sigma[3] * (np.abs(Z[3]) ** beta_active)
        ])

        # Create DataFrame
        df = pd.DataFrame({
            'Z': Z,
            'sigma': sigma,
            'h': h,
            'daily_value_added': daily_value
        })

        # Fit model by activity
        model = MarketImpactModel()
        model.df = df
        results = model.fit_by_activity()

        # Assertions for inactive regime
        self.assertAlmostEqual(results['eta_inactive'], eta_inactive, places=2)
        self.assertAlmostEqual(results['beta_inactive'], beta_inactive, places=2)

        # Assertions for active regime
        self.assertAlmostEqual(results['eta_active'], eta_active, places=2)
        self.assertAlmostEqual(results['beta_active'], beta_active, places=2)

    def test_wfb3875(self):
        # Well-conditioned data to ensure curve_fit succeeds
        Z = np.array([1.0, 2.0, 3.0, 4.0])
        sigma = np.array([0.5, 0.5, 0.5, 0.5])
        eta_true = 0.2
        beta_true = 0.6
        h = eta_true * sigma * (np.abs(Z) ** beta_true) * np.sign(Z)

        # Initialize model with correct data
        model = MarketImpactModel()
        model.df = pd.DataFrame({'Z': Z, 'sigma': sigma, 'h': h})
        model.eta_hat = eta_true
        model.beta_hat = beta_true
        model.seed = 42

        # Run bootstrap
        model.bootstrap(n_boot=100)

        # Check results
        if model.betas.std(ddof=1) == 0:
            self.assertTrue(np.isinf(model.t_beta))
        else:
            self.assertTrue(np.isfinite(model.t_beta))

    def test_ghy2764(self):
        # Poorly conditioned data: all Z values are zero, making curve_fit ill-posed
        Z_bad = np.array([0.0, 0.0, 0.0, 0.0])
        sigma_bad = np.array([0.5, 0.5, 0.5, 0.5])
        h_bad = np.array([0.0, 0.0, 0.0, 0.0])

        # Initialize model with bad data
        model_bad = MarketImpactModel()
        model_bad.df = pd.DataFrame({'Z': Z_bad, 'sigma': sigma_bad, 'h': h_bad})
        model_bad.eta_hat = 0.1
        model_bad.beta_hat = 0.5
        model_bad.seed = 123

        # Run bootstrap with low n_boot for speed
        model_bad.bootstrap(n_boot=10)

        # All fits should fail → result arrays remain empty
        self.assertTrue(np.allclose(model_bad.etas, model_bad.eta_hat))
        self.assertTrue(np.allclose(model_bad.betas, model_bad.beta_hat))

    def test_isu2725(self):
        # Simulated data
        np.random.seed(0)
        n = 100
        x1 = np.random.normal(size=n)
        x2 = np.random.normal(size=n)
        eps = np.random.normal(scale=1 + 0.5 * x1 ** 2, size=n)  # Heteroskedastic error
        y = 1 + 2 * x1 + 3 * x2 + eps

        df = pd.DataFrame({'x1': x1, 'x2': x2, 'h': y})

        model = MarketImpactModel()
        # Run White's test
        result = model.run_whites_test(df, x_cols=['x1', 'x2'], y_col='h', print_result=False)

        # Check results
        self.assertAlmostEqual(result["Test Statistic"], 25.368, delta=0.1)
        self.assertAlmostEqual(result["p-value"], 0.000, places=2)
        self.assertAlmostEqual(result["F-Statistic"], 6.390, places=2)
        self.assertAlmostEqual(result["F p-value"], 0.000, places=2)

if __name__ == '__main__':
    unittest.main()
