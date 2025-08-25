# models/ridge_lasso_regression.py

import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evaluation.regression_metrics import RegressionMetrics

class RidgeRegression:
    """
    Ridge Regression (L2 Regularization) with Gradient Descent.
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000, alpha=1.0):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.alpha = alpha  # Regularization parameter
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            # Add L2 penalty to the gradient of the weights
            dw = (1/n_samples) * (np.dot(X.T, (y_predicted - y)) + 2 * self.alpha * self.weights)
            db = (1/n_samples) * np.sum(y_predicted - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

class LassoRegression:
    """
    Lasso Regression (L1 Regularization) with Gradient Descent.
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000, alpha=1.0):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            # Add L1 penalty to the gradient of the weights
            dw = (1/n_samples) * (np.dot(X.T, (y_predicted - y)) + self.alpha * np.sign(self.weights))
            db = (1/n_samples) * np.sum(y_predicted - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

if __name__ == '__main__':
    data_path = 'data/generated_csv/ridge_lasso_data.csv'
    data = pd.read_csv(data_path)
    X = data.drop('Target', axis=1).values
    y = data['Target'].values
    X_scaled = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # Ridge
    ridge_model = RidgeRegression(alpha=0.1, n_iterations=2000)
    ridge_model.fit(X_scaled, y)
    ridge_predictions = ridge_model.predict(X_scaled)
    print("\n--- Ridge Regression ---")
    print("Ridge Weights:", np.round(ridge_model.weights, 2))
    ridge_eval = RegressionMetrics(y, ridge_predictions)
    print(f"R-squared: {ridge_eval.r_squared():.4f}")

    # Lasso
    lasso_model = LassoRegression(alpha=0.1, n_iterations=2000, learning_rate=0.005)
    lasso_model.fit(X_scaled, y)
    lasso_predictions = lasso_model.predict(X_scaled)
    print("\n--- Lasso Regression ---")
    print("Lasso Weights:", np.round(lasso_model.weights, 2))
    lasso_eval = RegressionMetrics(y, lasso_predictions)
    print(f"R-squared: {lasso_eval.r_squared():.4f}")