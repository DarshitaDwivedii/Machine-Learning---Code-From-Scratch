# models/logistic_regression.py

import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evaluation.classification_metrics import ClassificationMetrics

class LogisticRegression:
    """
    Logistic Regression for binary classification using Gradient Descent.
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)
            
            # Compute gradients for the log-loss function
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        """Predicts class labels (0 or 1) for new data."""
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted_proba = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted_proba]
        return np.array(y_predicted_cls)

if __name__ == '__main__':
    data_path = 'data/generated_csv/logistic_regression_data.csv'
    data = pd.read_csv(data_path)
    X = data.drop('Target', axis=1).values
    y = data['Target'].values
    
    # Initialize and train the model
    model = LogisticRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X, y)
    predictions = model.predict(X)
    
    # Evaluate the model
    print("\n--- Logistic Regression Evaluation ---")
    evaluator = ClassificationMetrics(y, predictions)
    print("Confusion Matrix:\n", evaluator.confusion_matrix())
    print(f"Accuracy: {evaluator.accuracy():.4f}")
    print(f"Precision: {evaluator.precision():.4f}")
    print(f"Recall: {evaluator.recall():.4f}")
    print(f"F1 Score: {evaluator.f1_score():.4f}")