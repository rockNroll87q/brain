"""
Code by Austin Dibble 2024, University of Glasgow
Copyright (c) 2024 Austin Dibble

This code file is for unit testing the metric functions from metrics.py
"""

import unittest

import numpy as np
import pandas as pd

from brain.metrics import (
    BasicMetricsWrapper,
    LongitudinalMetricsWrapper,
    categorize,
)


# test with `python -m unittest` from brain root directory
class TestMetricsFunctions(unittest.TestCase):
    """Test for metrics utilities."""

    def setUp(self):
        """Set up random fake values."""
        # Sample 100 values from a normal distribution for testing
        np.random.seed(184)  # For reproducibility
        self.y_true = np.random.normal(loc=50, scale=10, size=100)  # Mean=50, Std=10
        self.y_pred = self.y_true + np.random.normal(loc=0, scale=5, size=100)  # Adding some noise
        self.df = pd.DataFrame({"y_true": self.y_true, "y_pred": self.y_pred})

        # For Longitudinal Metrics
        self.y1 = np.random.normal(loc=50, scale=10, size=100)
        self.y2 = self.y1 + np.random.normal(loc=5, scale=2, size=100)  # Assuming some progression
        self.y_pred1 = self.y1 + np.random.normal(loc=0, scale=5, size=100)
        self.y_pred2 = self.y2 + np.random.normal(loc=0, scale=5, size=100)
        self.long_df = pd.DataFrame(
            {
                "y1": self.y1,
                "y2": self.y2,
                "y_pred1": self.y_pred1,
                "y_pred2": self.y_pred2,
            }
        )

    def test_mae(self):
        """Test MAE."""
        wrapper = BasicMetricsWrapper(y_true=self.y_true, y_pred=self.y_pred)
        expected_mae = np.mean(np.abs(self.y_pred - self.y_true))
        self.assertAlmostEqual(wrapper.mae(), expected_mae)

        wrapper_df = BasicMetricsWrapper(df=self.df, y_true='y_true', y_pred='y_pred')
        self.assertAlmostEqual(wrapper_df.mae(), expected_mae)

    def test_me(self):
        """Test ME."""
        wrapper = BasicMetricsWrapper(y_true=self.y_true, y_pred=self.y_pred)
        expected_me = np.mean(self.y_pred - self.y_true)
        self.assertAlmostEqual(wrapper.me(), expected_me)

        wrapper_df = BasicMetricsWrapper(df=self.df, y_true='y_true', y_pred='y_pred')
        self.assertAlmostEqual(wrapper_df.me(), expected_me)

    def test_mse(self):
        """Test MSE."""
        wrapper = BasicMetricsWrapper(y_true=self.y_true, y_pred=self.y_pred)
        expected_mse = np.mean((self.y_pred - self.y_true) ** 2)
        self.assertAlmostEqual(wrapper.mse(), expected_mse)

        wrapper_df = BasicMetricsWrapper(df=self.df, y_true='y_true', y_pred='y_pred')
        self.assertAlmostEqual(wrapper_df.mse(), expected_mse)

    def test_rmse(self):
        """Test RMSE."""
        wrapper = BasicMetricsWrapper(y_true=self.y_true, y_pred=self.y_pred)
        expected_rmse = np.sqrt(np.mean((self.y_pred - self.y_true) ** 2))
        self.assertAlmostEqual(wrapper.rmse(), expected_rmse)

        wrapper_df = BasicMetricsWrapper(df=self.df, y_true='y_true', y_pred='y_pred')
        self.assertAlmostEqual(wrapper_df.rmse(), expected_rmse)

    def test_r_squared(self):
        """Test R^2."""
        wrapper = BasicMetricsWrapper(y_true=self.y_true, y_pred=self.y_pred)
        expected_r2 = 1 - (
            (np.sum((self.y_true - self.y_pred) ** 2))
            / (np.sum((self.y_true - np.mean(self.y_true)) ** 2))
        )
        self.assertAlmostEqual(wrapper.r_squared(), expected_r2)
        
        wrapper_df = BasicMetricsWrapper(df=self.df, y_true='y_true', y_pred='y_pred')
        self.assertAlmostEqual(wrapper_df.r_squared(), expected_r2)

    def test_pearsons_r(self):
        """Test r."""
        wrapper = BasicMetricsWrapper(y_true=self.y_true, y_pred=self.y_pred)
        expected_pearson_r = np.corrcoef(self.y_true, self.y_pred)[0, 1]
        self.assertAlmostEqual(wrapper.pearsons_r(), expected_pearson_r)

        wrapper_df = BasicMetricsWrapper(df=self.df, y_true='y_true', y_pred='y_pred')
        self.assertAlmostEqual(wrapper_df.pearsons_r(), expected_pearson_r)

    def test_maxMAE(self):
        """Test max(MAE)."""
        wrapper = BasicMetricsWrapper(y_true=self.y_true, y_pred=self.y_pred)

        # Compute expected max MAE manually
        categories = np.arange(15, 86, 10)
        cat = categorize(self.y_true, bin=categories, int_bin=True)

        # Calculate the max MAE across all categories
        expected_max_mae = max(
            np.mean(np.abs(self.y_pred[cat == category] - self.y_true[cat == category]))
            for category in np.unique(cat)
        )

        self.assertAlmostEqual(wrapper.maxMAE(categories), expected_max_mae)

        wrapper_df = BasicMetricsWrapper(df=self.df, y_true='y_true', y_pred='y_pred')
        self.assertAlmostEqual(wrapper_df.maxMAE(categories), expected_max_mae)

    def test_longitudinal_mde(self):
        """Test longitudinal; MDE."""
        wrapper = LongitudinalMetricsWrapper(
            y1=self.y1, y2=self.y2, y_pred1=self.y_pred1, y_pred2=self.y_pred2
        )
        expected_mde = np.mean((self.y_pred2 - self.y_pred1) - (self.y2 - self.y1))
        self.assertAlmostEqual(wrapper.mde(), expected_mde)

        wrapper_df = LongitudinalMetricsWrapper(
            df=self.long_df,
            y1='y1',
            y2='y2',
            y_pred1='y_pred1',
            y_pred2='y_pred2'
        )
        self.assertAlmostEqual(wrapper_df.mde(), expected_mde)

    def test_longitudinal_mde_sd(self):
        """Test longitudinal; MDE-SD."""
        wrapper = LongitudinalMetricsWrapper(
            y1=self.y1, y2=self.y2, y_pred1=self.y_pred1, y_pred2=self.y_pred2
        )
        expected_mde_sd = np.std((self.y_pred2 - self.y_pred1) - (self.y2 - self.y1))
        self.assertAlmostEqual(wrapper.mde_sd(), expected_mde_sd)

        wrapper_df = LongitudinalMetricsWrapper(
            df=self.long_df,
            y1='y1',
            y2='y2',
            y_pred1='y_pred1',
            y_pred2='y_pred2'
        )
        self.assertAlmostEqual(wrapper_df.mde_sd(), expected_mde_sd)

    def test_longitudinal_made(self):
        """Test longitudinal; MAdE."""
        wrapper = LongitudinalMetricsWrapper(
            y1=self.y1, y2=self.y2, y_pred1=self.y_pred1, y_pred2=self.y_pred2
        )
        expected_made = np.mean(
            np.abs((self.y_pred2 - self.y_pred1) - (self.y2 - self.y1))
        )
        self.assertAlmostEqual(wrapper.made(), expected_made)

        wrapper_df = LongitudinalMetricsWrapper(
            df=self.long_df,
            y1='y1',
            y2='y2',
            y_pred1='y_pred1',
            y_pred2='y_pred2'
        )
        self.assertAlmostEqual(wrapper_df.made(), expected_made)

    def test_longitudinal_made_sd(self):
        """Test longitudinal; MAdE-SD."""
        wrapper = LongitudinalMetricsWrapper(
            y1=self.y1, y2=self.y2, y_pred1=self.y_pred1, y_pred2=self.y_pred2
        )
        expected_made_sd = np.std(
            np.abs((self.y_pred2 - self.y_pred1) - (self.y2 - self.y1))
        )
        self.assertAlmostEqual(wrapper.made_sd(), expected_made_sd)

        wrapper_df = LongitudinalMetricsWrapper(
            df=self.long_df,
            y1='y1',
            y2='y2',
            y_pred1='y_pred1',
            y_pred2='y_pred2'
        )
        self.assertAlmostEqual(wrapper_df.made_sd(), expected_made_sd)

    def test_longitudinal_maxMAdE(self):
        """Test longitudinal; max(MAdE)."""
        y_cat = self.y1  # Assuming y_cat corresponds to y1 in longitudinal data
        y = self.y2 - self.y1
        y_pred = self.y_pred2 - self.y_pred1

        wrapper = LongitudinalMetricsWrapper(
            y1=self.y1, y2=self.y2, y_pred1=self.y_pred1, y_pred2=self.y_pred2
        )

        # Compute expected max MAdE manually
        categories = np.arange(15, 86, 10)
        cat = categorize(y_cat, bin=categories, int_bin=True)

        # Calculate the max MAdE across all categories
        expected_max_made = max(
            np.mean(np.abs(y_pred[cat == category] - y[cat == category]))
            for category in np.unique(cat)
        )

        self.assertAlmostEqual(
            wrapper.maxMAdE(y_cat=y_cat, y=y, y_pred=y_pred, categories=categories),
            expected_max_made,
        )

        wrapper_df = LongitudinalMetricsWrapper(
            df=self.long_df,
            y1='y1',
            y2='y2',
            y_pred1='y_pred1',
            y_pred2='y_pred2'
        )
        self.assertAlmostEqual(wrapper_df.maxMAdE(
            y_cat=y_cat, y=y, y_pred=y_pred, categories=categories
            ), expected_max_made)

    def test_longitudinal_k(self):
        """Test longitudinal; k."""

        wrapper = LongitudinalMetricsWrapper(
            y1=self.y1, y2=self.y2, y_pred1=self.y_pred1, y_pred2=self.y_pred2
        )
        expected_k = (self.y_pred2 - self.y_pred1) / (self.y2 - self.y1)
        np.testing.assert_array_almost_equal(wrapper.k(), expected_k)

        wrapper_df = LongitudinalMetricsWrapper(
            df=self.long_df,
            y1='y1',
            y2='y2',
            y_pred1='y_pred1',
            y_pred2='y_pred2'
        )
        np.testing.assert_array_almost_equal(wrapper_df.k(), expected_k)


if __name__ == "__main__":
    unittest.main()
