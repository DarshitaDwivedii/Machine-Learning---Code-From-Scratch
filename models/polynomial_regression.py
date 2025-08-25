# models/polynomial_regression.py

import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evaluation.regression_metrics import RegressionMetrics
from models.multiple_regression import MultipleLinearRegression

class PolynomialRegression:
    """
    Polynomial Regression model built on top of MultipleLinearRegression.
    """
    def __init__(self, degree, learning_rate=0.01, n_iterations=1000):
        self.degree = degree
        # This model is a special case of multiple linear regression
        self.linear_regression = MultipleLinearRegression(learning_rate, n_iterations)

    def _transform_features(self, X):
        """Transforms the input features into polynomial features."""
        # Start with a column of ones for the bias term
        X_poly = np.ones((X.shape[0], 1))
        for d in range(1, self.degree + 1):
            X_poly = np.c_[X_poly, X**d]
        return X_poly

    def fit(self, X, y):
        """Fits the model by transforming features and then training."""
        X_poly = self._transform_features(X)
        self.linear_regression.fit(X_poly, y)

    def predict(self, X):
        """Makes predictions by transforming features and then predicting."""
        X_poly = self._transform_features(X)
        return self.linear_regression.predict(X_poly)

if __name__ == '__main__':
    data_path = 'data/generated_csv/polynomial_regression_data.csv'
    data = pd.read_csv(data_path)
    X = data[['Feature']].values
    y = data['Target'].values.flatten()

    # Initialize and train the model
    model = PolynomialRegression(degree=2, learning_rate=0.001, n_iterations=2000)
    model.fit(X, y)
    predictions = model.predict(X)
    
    print("Polynomial Regression model (degree=2) trained.")
    
    # Evaluate the model
    print("\n--- Polynomial Regression Evaluation ---")
    evaluator = RegressionMetrics(y, predictions)
    print(f"Mean Squared Error (MSE): {evaluator.mean_squared_error():.4f}")
    print(f"R-squared: {evaluator.r_squared():.4f}")