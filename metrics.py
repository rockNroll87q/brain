"""
Original code from BASE, licensed under the MIT License.
Copyright (c) 2022 Lara

Modifications by Austin Dibble 2024, University of Glasgow
Copyright (c) 2024 Austin Dibble
"""

"""
== Original MIT License Here ==
MIT License

Copyright (c) 2022 Lara

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import math

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Copied from src.auxiliary from original source repository
def categorize(
    num_vector,
    bin=(18, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 100),
    int_bin=False,
):
    """
    Categorize each element of a numerical vector into predefined bins.

    Parameters:
    :param num_vector (array_like, float): A numpy.ndarray (or a structure that can be converted to it) containing
    numerical data to be categorized.
    :param bin (tuple of int/float, optional): A tuple defining the bin edges for categorization. Default is
    (18, 20, 25, 30, 35, ..., 90, 100).
    :param int_bin (bool, optional): If True, categories are returned as integers representing bin indices;
    if False, categories are returned as strings showing bin ranges. Default is False.

    Returns:
    :return: A numpy.ndarray containing the categorized data. The data type of the array is either 'int'
    (if int_bin is True) or 'str' (if int_bin is False).
    """
    bin = np.array(bin, dtype=np.float32)
    num_vector = np.array(num_vector)

    if not np.all(np.sort(bin) == bin):
        raise ValueError("Categories should be in increasing order")
    if len(bin) < 2:
        warnings.warn("Vector remains unchanged; Expected len(bin)>=2.")
        return num_vector

    # Check for out of range values
    if np.any(num_vector < bin[0]) or np.any(num_vector > bin[-1]):
        raise ValueError(f"Values should be within the range [{bin[0]}, {bin[-1]}]")

    # Assign bin
    category_labels = [f"[{int(cat1)},{int(cat2)})" for cat1, cat2 in zip(bin, bin[1:])]
    category_labels[-1] = f"[{int(bin[-2])},{int(bin[-1])}]"

    if int_bin:
        # skip the first left bin edge bin[0] since digitize assumes (-Inf, bin[0])
        # skip the right most bin[-1], to include values x=bin[-1] to the last interval [bin[-2], bin[-1]]
        categorized = np.digitize(num_vector, bin[1:-1], right=False)
    else:
        categorized = np.array(
            [category_labels[i] for i in np.digitize(num_vector, bin[1:-1])]
        )
    return np.array(categorized)


# == Begin modified code than that from original MIT-licensed author Lara in order to make it use numpy, not pandas ==
def maxMAE(
    y: np.ndarray,
    y_pred: np.ndarray,
    categories=np.append(
        np.append(np.array([18]), np.arange(25, 86, 10)), np.array([100])
    ),
):
    r"""
    Compute max MAE over given categories;
    """
    # categorize values of y
    cat = categorize(y, bin=categories, int_bin=True)

    mae_max = list()
    for category in np.unique(cat):
        y_i = y[cat == category]
        y_pred_i = y_pred[cat == category]
        mae_i = np.mean(np.abs(y_pred_i - y_i))
        # append to list
        mae_max.append(mae_i)
    return max(mae_max)


def maxMAdE(
    y_cat: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray,
    categories=np.append(
        np.append(np.array([18]), np.arange(25, 86, 10)), np.array([100])
    ),
):
    r"""
    Compute max MAdE (delta error) over given categories; Subjects are categorized based on y_cat. Max error values
     is computed based on y and y_pred values. y_cat is usually starting age ('y_1' in the original code)
    """
    # categorize values of y
    cat = categorize(y_cat, bin=categories, int_bin=True)

    mae_max = list()
    for category in np.unique(cat):
        y_i = y[cat == category]
        y_pred_i = y_pred[cat == category]
        mae_i = np.mean(np.abs(y_pred_i - y_i))
        # append to list
        mae_max.append(mae_i)
    return max(mae_max)


def pearsons_r(y: np.ndarray, y_pred: np.ndarray):
    r"""
    Compute Pearson's correlation between two columns;
    """
    y_np = y
    y_pred_np = y_pred
    corr_mat = np.corrcoef(y_np, y_pred_np)
    return corr_mat[0, 1]  # y_true, y_pred


def RMSE(y: np.ndarray, y_pred: np.ndarray):
    r"""
    RMSE between true and predicted value;
    """
    return math.sqrt(mean_squared_error(y, y_pred))


def R_squared(y: np.ndarray, y_pred: np.ndarray):
    r"""
    R² between true and predicted value;
    """
    return r2_score(y, y_pred)


# == Code below here was completely added by Austin Dibble ==
from sklearn.metrics import mean_absolute_error, mean_squared_error


class BasicMetricsWrapper:
    def __init__(
        self, y_true=None, y_pred=None, df=None, y_true_col=None, y_pred_col=None
    ):
        if all(isinstance(arg, str) for arg in [y_true_col, y_pred_col]):
            if df is None:
                raise ValueError(
                    "If y_true_col and y_pred_col string column names, df must be provided."
                )

            self.y_true = df[y_true_col].values
            self.y_pred = df[y_pred_col].values

        elif all(isinstance(arg, np.ndarray) for arg in [y_true, y_pred]):
            if len(y_true) != len(y_pred):
                raise ValueError("y_true and y_pred must have the same length.")

            self.y_true = y_true
            self.y_pred = y_pred
        else:
            raise ValueError(
                "Either (y_true and y_pred as numpy arrays) or (df and column names as strings) must be provided."
            )

        self.default_bins = np.append(
            np.append(np.array([18]), np.arange(25, 86, 10)), np.array([100])
        )

    def mae(self):
        return mean_absolute_error(self.y_true, self.y_pred)

    def me(self):
        return np.mean(self.y_pred - self.y_true)

    def mse(self):
        return mean_squared_error(self.y_true, self.y_pred)

    def rmse(self):
        return RMSE(self.y_true, self.y_pred)

    def r_squared(self):
        return R_squared(self.y_true, self.y_pred)

    def pearsons_r(self):
        return pearsons_r(self.y_true, self.y_pred)

    def maxMAE(self):
        return maxMAE(self.y_true, self.y_pred, categories=self.default_bins)

    def calculate_metrics(self):
        metrics = {
            "MAE": self.mae(),
            "MSE": self.mse(),
            "RMSE": self.rmse(),
            "r_squared": self.r_squared(),
            "pearsons_r": self.pearsons_r(),
            "maxMAE": self.maxMAE(),
        }
        return pd.DataFrame([metrics])


class LongitudinalMetricsWrapper:
    def __init__(self, y1, y2, y_pred1, y_pred2, df=None):
        if all(isinstance(arg, str) for arg in [y1, y2, y_pred1, y_pred2]):
            if df is None:
                raise ValueError(
                    "If y1, y2, y_pred1, and y_pred2 are column names, df must be provided."
                )
            self.y1 = df[y1].values
            self.y2 = df[y2].values
            self.y_pred1 = df[y_pred1].values
            self.y_pred2 = df[y_pred2].values

        elif all(isinstance(arg, np.ndarray) for arg in [y1, y2, y_pred1, y_pred2]):
            if len(y1) != len(y2) or len(y1) != len(y_pred1) or len(y1) != len(y_pred2):
                raise ValueError("All numpy arrays must have the same length.")
            self.y1 = y1
            self.y2 = y2
            self.y_pred1 = y_pred1
            self.y_pred2 = y_pred2

        else:
            raise ValueError(
                "y1, y2, y_pred1, and y_pred2 must be either all strings or all numpy arrays."
            )

        self.default_bins = np.append(
            np.append(np.array([18]), np.arange(25, 86, 10)), np.array([100])
        )

        self.y_delta = self.y2 - self.y1
        self.y_pred_delta = self.y_pred2 - self.y_pred1
        self.y_pred_delta_diff = self.y_pred_delta - self.y_delta
        self.y_pred_delta_diff_abs = np.abs(self.y_pred_delta_diff)

    def mde(self):
        return np.mean(self.y_pred_delta_diff)

    def mde_sd(self):
        return np.std(self.y_pred_delta_diff)

    def made(self):
        return np.mean(self.y_pred_delta_diff_abs)

    def made_sd(self):
        return np.std(self.y_pred_delta_diff_abs)

    def maxMAdE_10y(self):
        return self.maxMAdE(
            self.y1, self.y_delta, self.y_pred_delta, categories=self.default_bins
        )

    def maxMAdE(self, y_cat, y, y_pred, categories):
        return maxMAdE(y_cat, y, y_pred, categories=categories)

    def k(self):
        return self.y_pred_delta / self.y_delta

    def calculate_metrics(self):
        metrics = {
            "MdE": self.mde(),
            "MdEsd": self.mde_sd(),
            "MAdE": self.made(),
            "MAdEsd": self.made_sd(),
            "maxMAdE_10y": self.maxMAdE_10y(),
            "k": self.k(),
        }
        return pd.DataFrame([metrics])


# Example usage with a DataFrame:
# Assuming df is your DataFrame and it contains the columns: 'y1', 'y2', 'y_pred1', 'y_pred2'
# metrics = LongitudinalMetricsWrapper(df=df, y1_col='y1', y2_col='y2', y_pred1_col='y_pred1', y_pred2_col='y_pred2')
# print(metrics.paper_table2())

# Example usage with numpy arrays:
# y1 = np.array([...])
# y2 = np.array([...])
# y_pred1 = np.array([...])
# y_pred2 = np.array([...])
# metrics = LongitudinalMetricsWrapper(y1=y1, y2=y2, y_pred1=y_pred1, y_pred2=y_pred2)
# print(metrics.paper_table2())
