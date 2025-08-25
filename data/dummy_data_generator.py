# data/dummy_data_generator.py

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import os

class DataGenerator:
    """
    A class to generate and save dummy datasets for various ML models.
    """
    def __init__(self, random_state=42, base_path="data/generated_csv"):
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    def _get_path(self, filename):
        """Constructs the full file path for saving the CSV."""
        return os.path.join(self.base_path, filename)

    def generate_simple_regression_data(self, n_samples=100, noise=10, filename="simple_regression_data.csv"):
        """Generates data for simple linear regression."""
        filepath = self._get_path(filename)
        X = 2 * np.random.rand(n_samples, 1)
        y = 4 + 3 * X + np.random.randn(n_samples, 1) * noise
        df = pd.DataFrame(data=np.hstack([X, y]), columns=['Feature', 'Target'])
        df.to_csv(filepath, index=False)
        print(f"Generated and saved simple regression data to {filepath}")

    def generate_multiple_regression_data(self, n_samples=100, n_features=3, noise=20, filename="multiple_regression_data.csv"):
        """Generates data for multiple linear regression."""
        filepath = self._get_path(filename)
        X = np.random.rand(n_samples, n_features) * 10
        true_betas = np.array([2.5, -3.2, 1.8])
        y = X @ true_betas + 5 + np.random.randn(n_samples) * noise
        df = pd.DataFrame(data=X, columns=[f'Feature_{i+1}' for i in range(n_features)])
        df['Target'] = y
        df.to_csv(filepath, index=False)
        print(f"Generated and saved multiple regression data to {filepath}")

    def generate_polynomial_regression_data(self, n_samples=100, noise=1.5, filename="polynomial_regression_data.csv"):
        """Generates data for polynomial regression."""
        filepath = self._get_path(filename)
        X = 6 * np.random.rand(n_samples, 1) - 3
        y = 0.5 * X**2 + X + 2 + np.random.randn(n_samples, 1) * noise
        df = pd.DataFrame(data=np.hstack([X, y]), columns=['Feature', 'Target'])
        df.to_csv(filepath, index=False)
        print(f"Generated and saved polynomial regression data to {filepath}")

    def generate_ridge_lasso_data(self, n_samples=100, n_features=10, noise=15, filename="ridge_lasso_data.csv"):
        """Generates data suitable for Ridge and Lasso regression."""
        filepath = self._get_path(filename)
        X = np.random.rand(n_samples, n_features) * 15
        X[:, 2] = X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 2
        true_betas = np.array([3, 1.5, 0, 0, 2, 0, 0, 0, 0, 0])
        y = X @ true_betas + 7 + np.random.randn(n_samples) * noise
        df = pd.DataFrame(data=X, columns=[f'Feature_{i+1}' for i in range(n_features)])
        df['Target'] = y
        df.to_csv(filepath, index=False)
        print(f"Generated and saved Ridge/Lasso regression data to {filepath}")

    def generate_logistic_regression_data(self, n_samples=200, n_features=2, filename="logistic_regression_data.csv"):
        """Generates data for logistic regression."""
        filepath = self._get_path(filename)
        X, y = make_classification(n_samples=n_samples, n_features=n_features, n_redundant=0,
                                   n_informative=2, random_state=self.random_state,
                                   n_clusters_per_class=1)
        df = pd.DataFrame(data=X, columns=[f'Feature_{i+1}' for i in range(n_features)])
        df['Target'] = y
        df.to_csv(filepath, index=False)
        print(f"Generated and saved logistic regression data to {filepath}")

    def generate_decision_tree_data(self, n_samples=200, n_features=2, filename="decision_tree_data.csv"):
        """Generates data for a decision tree classifier."""
        filepath = self._get_path(filename)
        X, y = make_classification(n_samples=n_samples, n_features=n_features, n_redundant=0,
                                   n_informative=2, random_state=self.random_state,
                                   n_clusters_per_class=2)
        df = pd.DataFrame(data=X, columns=[f'Feature_{i+1}' for i in range(n_features)])
        df['Target'] = y
        df.to_csv(filepath, index=False)
        print(f"Generated and saved decision tree data to {filepath}")

if __name__ == '__main__':
    generator = DataGenerator()
    generator.generate_simple_regression_data()
    generator.generate_multiple_regression_data()
    generator.generate_polynomial_regression_data()
    generator.generate_ridge_lasso_data()
    generator.generate_logistic_regression_data()
    generator.generate_decision_tree_data()