# ML Models from Scratch: OOP Implementation

This repository contains machine learning models implemented entirely from scratch using the professional Object-Oriented Programming (OOP) paradigm in Python. The project structure separates data generation, model logic, and evaluation metrics into modular, reusable components.

## 1. Project Structure

The project follows a clean separation of concerns:

```
ml_from_scratch/
|
├── data/
│   ├── generated_csv/   # All generated data files reside here.
│   └── dummy_data_generator.py # Script to create all datasets.
|
├── evaluation/
│   ├── classification_metrics.py # Accuracy, Precision, Recall, F1.
│   └── regression_metrics.py     # MSE, RMSE, R-squared.
|
└── models/
    ├── *regression_models.py
    └── *classification_models.py
```

## 2. Setup and Execution

### Prerequisites

You need the following libraries installed:
```bash
pip install numpy pandas scikit-learn
```

### Execution Steps

1. **Generate Data:** Run the generator script to create all necessary CSV files in the `data/generated_csv` directory.
   ```bash
   python data/dummy_data_generator.py
   ```
2. **Run Models:** Execute any model file directly to train the model, make predictions, and see the evaluation metrics.
   ```bash
   python models/simple_regression.py
   python models/logistic_regression.py
   # etc.
   ```

## 3. Regression Models

Regression models are used for predicting a continuous target variable.

### 3.1 Simple Linear Regression

| Dataset | Type |
| :--- | :--- |
| **simple_regression_data.csv** | 1 Feature (X), 1 Target (Y) |

#### Model Overview
Simple Linear Regression aims to find the best linear relationship between one independent variable ($X$) and one dependent variable ($Y$). Our implementation uses the **Ordinary Least Squares (OLS)** analytical solution to find the slope and intercept.

| Parameter | Description | Notes on Tuning |
| :--- | :--- | :--- |
| N/A | Analytical Solution | This method has no hyperparameters to tune; it finds the optimal solution mathematically. |

#### Performance (Sample Run)

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Intercept** | `6.1510` | The value of Y when X is 0. |
| **Slope** | `0.7011` | For every 1-unit increase in X, Y increases by 0.7011. |
| **MSE** | `80.6585` | Average squared error. High value indicates large noise/outliers. |
| **R-squared** | `0.0021` | The model explains only 0.21% of the variance in the target variable. This indicates the synthetic data generated has a very weak linear relationship, dominated by random noise. |

---

### 3.2 Multiple & Polynomial Regression

| Dataset | Type |
| :--- | :--- |
| **multiple_regression_data.csv** | 3 Features, 1 Target |
| **polynomial_regression_data.csv** | 1 Feature, 1 Target (with parabolic relationship) |

#### Model Overview
Both models use the same underlying **Gradient Descent** optimization algorithm to find weights for multiple features.

| Parameter | Description | Notes on Tuning |
| :--- | :--- | :--- |
| **`learning_rate` (lr)** | Step size taken during optimization. | **Critical:** Too high, the model may diverge (fail to converge). Too low, training is very slow. Typically start small (0.01 or 0.001). |
| **`n_iterations`** | Number of steps the optimizer takes. | Too low results in underfitting. Increase until loss stabilizes. |
| **`degree` (Poly)** | Highest power of the feature used. | Controls complexity. Higher degree risks severe **overfitting**. |

#### Performance (Sample Runs)

| Model | Metric | Value | Interpretation |
| :--- | :--- | :--- | :--- |
| **Multiple Regression** | MSE | `451.9773` | High error, indicative of large noise in the synthetic data. |
| | R-squared | `0.3085` | The linear model explains about 31% of the variance. |
| **Polynomial (Degree 2)** | MSE | `2.5427` | Significantly lower error than linear models, as expected. |
| | R-squared | `0.6265` | Explains over 62% of the variance, confirming the quadratic relationship in the data was successfully captured. |

---

### 3.3 Ridge and Lasso Regression (Regularization)

| Dataset | Type |
| :--- | :--- |
| **ridge_lasso_data.csv** | 10 Features (designed with some features being irrelevant/collinear) |

#### Model Overview
These are advanced linear models that add a penalty term to the cost function to prevent **overfitting** and handle collinearity.

| Model | Penalty | Advantage |
| :--- | :--- | :--- |
| **Ridge (L2)** | Sum of squared weights ($\sum \theta_i^2$) | Shrinks coefficients towards zero, great for mitigating multicollinearity. |
| **Lasso (L1)** | Sum of absolute weights ($\sum |\theta_i|$) | Shrinks coefficients *exactly* to zero, performing implicit **feature selection**. |

| Parameter | Description | Notes on Tuning |
| :--- | :--- | :--- |
| **`alpha` ($\lambda$)** | The strength of the regularization penalty. | **Crucial:** A higher `alpha` increases the penalty, shrinking weights more aggressively. If `alpha=0`, it reverts to standard linear regression. Tune this using cross-validation. |

#### Performance (Sample Runs)

| Model | Weights (Sample) | R-squared | Observation |
| :--- | :--- | :--- | :--- |
| **Ridge ($\alpha=0.1$)** | `[9.31, 2.49, 4.83, 0.36, 5.95, -1.15, ...]` | `0.4879` | All 10 weights are non-zero, though some are very small. |
| **Lasso ($\alpha=0.1$)** | `[8.71, 2.02, 5.56, 0.35, 5.97, -1.16, ...]` | `0.4874` | Weights are slightly different, but none were driven completely to zero at this `alpha` level. |

**Note:** For Lasso to demonstrate strong feature selection (setting coefficients to 0), a higher `alpha` might be required, or the underlying data must have less relevant features.

---

## 4. Classification Models

Classification models are used for predicting a discrete categorical label.

### 4.1 Logistic Regression

| Dataset | Type |
| :--- | :--- |
| **logistic_regression_data.csv** | 2 Features, Binary Target (0 or 1) |

#### Model Overview
Despite its name, Logistic Regression is a classification algorithm. It uses the **sigmoid function** to map the linear output into a probability (0 to 1), which is then thresholded (usually at 0.5) to determine the class. It uses **Gradient Descent** to minimize the cross-entropy (log-loss) cost function.

| Parameter | Description | Notes on Tuning |
| :--- | :--- | :--- |
| **`learning_rate` (lr)** | Step size for gradient updates. | Essential for convergence. If classification accuracy oscillates wildly, decrease this. |
| **`n_iterations`** | Training epochs. | Ensure this is high enough for the model to converge to its maximum accuracy. |

#### Performance (Sample Run)

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | `0.8450` | 84.5% of all predictions were correct. |
| **Precision** | `0.8350` | Of all instances predicted as 1, 83.5% were actually 1. (Focus on false positives). |
| **Recall** | `0.8600` | Of all actual 1s, the model correctly identified 86.0%. (Focus on false negatives). |
| **F1 Score** | `0.8473` | The harmonic mean of Precision and Recall. |
| **Confusion Matrix** | `[[83 17], [14 86]]` | **TN=83, FP=17, FN=14, TP=86** |

---

### 4.2 Decision Tree Classifier

| Dataset | Type |
| :--- | :--- |
| **decision_tree_data.csv** | 2 Features, Binary Target (0 or 1) |

#### Model Overview
Decision Trees recursively partition the feature space based on **Information Gain (Entropy)**. They are highly flexible non-linear models that do not require feature scaling.

| Parameter | Description | Importance for Overfitting |
| :--- | :--- | :--- |
| **`max_depth`** | Maximum number of splits allowed. | **Crucial:** Low depth leads to **underfitting**. High depth risks severe **overfitting** (memorizing the training data). |
| **`min_samples_split`** | Minimum number of samples required to split an internal node. | A higher value prevents the tree from creating splits based on single outliers, reducing complexity. |
| **`n_feats`** | Number of features to consider at each split (used for Random Forest functionality). | Controls randomness. If set to the total number of features, it acts as a standard Decision Tree. |

#### Performance (Sample Run)

| Metric | Value | Observation |
| :--- | :--- | :--- |
| **Accuracy** | `1.0000` | Perfect 100% accuracy. |
| **F1 Score** | `1.0000` | Perfect F1 score. |
| **Confusion Matrix** | `[[100 0], [0 100]]` | TN=100, FP=0, FN=0, TP=100 |

**Warning on Performance:** The 100% accuracy is typical when running a deep Decision Tree on the *exact same data* it was trained on (training accuracy). **This is the classic sign of severe overfitting (memorization).**

### How to Prevent Overfitting (Pruning)

To make the Decision Tree generalize better to new, unseen data, you must use **Pruning** by tuning the following parameters:

1.  **Reduce `max_depth`**: Set `max_depth` to a smaller value (e.g., 3 or 5) to force the tree to generalize broader rules instead of deep, specific rules.
2.  **Increase `min_samples_split`**: Forcing the tree to only split nodes that have a large number of samples prevents the tree from isolating outliers.