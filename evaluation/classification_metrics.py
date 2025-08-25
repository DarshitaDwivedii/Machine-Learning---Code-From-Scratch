# evaluation/classification_metrics.py

import numpy as np

class ClassificationMetrics:
    """
    A class to calculate common classification evaluation metrics for binary cases.
    """
    def __init__(self, y_true, y_pred):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.tn, self.fp, self.fn, self.tp = self._confusion_matrix_values()

    def _confusion_matrix_values(self):
        """Helper function to compute TN, FP, FN, TP."""
        tn = np.sum((self.y_true == 0) & (self.y_pred == 0))
        fp = np.sum((self.y_true == 0) & (self.y_pred == 1))
        fn = np.sum((self.y_true == 1) & (self.y_pred == 0))
        tp = np.sum((self.y_true == 1) & (self.y_pred == 1))
        return tn, fp, fn, tp

    def confusion_matrix(self):
        """Returns the confusion matrix as a 2x2 numpy array."""
        return np.array([[self.tn, self.fp], [self.fn, self.tp]])

    def accuracy(self):
        """Calculates accuracy."""
        total = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / total if total > 0 else 0.0

    def precision(self):
        """Calculates precision (positive predictive value)."""
        denominator = self.tp + self.fp
        return self.tp / denominator if denominator > 0 else 0.0

    def recall(self):
        """Calculates recall (sensitivity or true positive rate)."""
        denominator = self.tp + self.fn
        return self.tp / denominator if denominator > 0 else 0.0

    def f1_score(self):
        """Calculates the F1 score."""
        prec = self.precision()
        rec = self.recall()
        denominator = prec + rec
        return 2 * (prec * rec) / denominator if denominator > 0 else 0.0