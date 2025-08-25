# evaluation/regression_metrics.py

import numpy as np

class RegressionMetrics:
    """
    A class to calculate common regression evaluation metrics.
    """
    def __init__(self, y_true, y_pred):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)

    def mean_squared_error(self):
        """Calculates Mean Squared Error (MSE)."""
        return np.mean((self.y_true - self.y_pred) ** 2)

    def root_mean_squared_error(self):
        """Calculates Root Mean Squared Error (RMSE)."""
        return np.sqrt(self.mean_squared_error())

    def mean_absolute_error(self):
        """Calculates Mean Absolute Error (MAE)."""
        return np.mean(np.abs(self.y_true - self.y_pred))

    def r_squared(self):
        """Calculates R-squared (Coefficient of Determination)."""
        ss_total = np.sum((self.y_true - np.mean(self.y_true)) ** 2)
        ss_residual = np.sum((self.y_true - self.y_pred) ** 2)
        if ss_total == 0:
            return 1.0 if ss_residual == 0 else 0.0
        return 1 - (ss_residual / ss_total)