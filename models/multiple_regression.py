# models/multiple_regression.py

import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evaluation.regression_metrics import RegressionMetrics

class MultipleLinearRegression:
    """
    A class for Multiple Linear Regression using Gradient Descent.
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Fits the multiple linear regression model using gradient descent.
        X: array-like, shape (n_samples, n_features)
        y: array-like, shape (n_samples,)
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        """Makes predictions using the trained model."""
        return np.dot(X, self.weights) + self.bias

if __name__ == '__main__':
    data_path = 'data/generated_csv/multiple_regression_data.csv'
    data = pd.read_csv(data_path)
    X = data.drop('Target', axis=1).values
    y = data['Target'].values
    
    # Feature scaling is important for gradient descent
    X_scaled = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # Initialize and train the model
    model = MultipleLinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X_scaled, y)
    predictions = model.predict(X_scaled)

    print(f"Bias (Intercept): {model.bias:.4f}")
    print(f"Weights (Slopes): {np.round(model.weights, 4)}")

    # Evaluate the model
    print("\n--- Multiple Regression Evaluation ---")
    evaluator = RegressionMetrics(y, predictions)
    print(f"Mean Squared Error (MSE): {evaluator.mean_squared_error():.4f}")
    print(f"R-squared: {evaluator.r_squared():.4f}")