# models/simple_regression.py

import numpy as np
import pandas as pd
import sys
import os

# Add the project's root directory to the Python path to allow imports from other directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evaluation.regression_metrics import RegressionMetrics

class SimpleLinearRegression:
    """
    A class for Simple Linear Regression using the analytical solution (Ordinary Least Squares).
    """
    def __init__(self):
        self.slope = None
        self.intercept = None

    def fit(self, X, y):
        """
        Fits the linear regression model.
        X: array-like, shape (n_samples, 1)
        y: array-like, shape (n_samples,)
        """
        X_flat = X.flatten()
        mean_x, mean_y = np.mean(X_flat), np.mean(y)

        # Calculate slope (m) and intercept (b) using OLS formulas
        numerator = np.sum((X_flat - mean_x) * (y - mean_y))
        denominator = np.sum((X_flat - mean_x)**2)

        self.slope = numerator / denominator
        self.intercept = mean_y - (self.slope * mean_x)

    def predict(self, X):
        """
        Makes predictions using the trained model.
        X: array-like, shape (n_samples, 1)
        """
        return self.slope * X.flatten() + self.intercept

if __name__ == '__main__':
    # Define the path to the data file relative to the project root
    data_path = 'data/generated_csv/simple_regression_data.csv'

    # Load data
    data = pd.read_csv(data_path)
    X = data[['Feature']].values
    y = data['Target'].values.flatten()

    # Initialize and train the model
    model = SimpleLinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)

    print(f"Intercept: {model.intercept:.4f}")
    print(f"Slope: {model.slope:.4f}")

    # Evaluate the model
    print("\n--- Simple Regression Evaluation ---")
    evaluator = RegressionMetrics(y, predictions)
    print(f"Mean Squared Error (MSE): {evaluator.mean_squared_error():.4f}")
    print(f"Root Mean Squared Error (RMSE): {evaluator.root_mean_squared_error():.4f}")
    print(f"R-squared: {evaluator.r_squared():.4f}")