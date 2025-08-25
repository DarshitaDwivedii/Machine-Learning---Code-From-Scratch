# ML Models from Scratch: An In-Depth OOP Implementation

This repository contains a comprehensive suite of machine learning models implemented from scratch in Python, following a professional Object-Oriented Programming (OOP) paradigm. The project is structured to be highly modular, separating the logic for data generation, model implementation, and evaluation.

This document serves as an in-depth guide to each model, explaining its theoretical underpinnings, practical implementation, and an analysis of its performance on the generated datasets.

## Project Structure

The project follows a clean separation of concerns for maximum clarity and scalability:

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
    ├── simple_regression.py
    ├── multiple_regression.py
    ├── polynomial_regression.py
    ├── ridge_lasso_regression.py
    ├── logistic_regression.py
    ├── decision_tree_nodes.py
    └── decision_tree_classifier.py
```

## Setup and Execution

### Prerequisites

You need the following libraries installed:
```bash
pip install numpy pandas scikit-learn
```

### Execution Steps

1. **Generate Data:** Run the generator script once to create all necessary CSV files.
   ```bash
   python data/dummy_data_generator.py
   ```
2. **Run Models:** Execute any model file to see it train and evaluate.
   ```bash
   python models/multiple_regression.py
   ```

---
---

## Model Deep Dive: Regression

### 1. Simple Linear Regression

#### Model Explanation
Simple Linear Regression is the most fundamental regression algorithm. It assumes a linear relationship between a single independent variable (feature, $X$) and a continuous dependent variable (target, $Y$). The goal is to find the optimal straight line, represented by the equation $Y = \beta_0 + \beta_1 X$, that best fits the data points. "Best fit" is defined as the line that minimizes the sum of the squared vertical distances (residuals) from each data point to the line. This implementation uses the **Ordinary Least Squares (OLS)** method, an analytical approach that calculates the optimal intercept ($\beta_0$) and slope ($\beta_1$) directly without iterative optimization.

#### Advantages
-   **Highly Interpretable:** The slope represents the change in Y for a one-unit change in X, and the intercept is the value of Y when X is zero.
-   **Fast and Efficient:** The analytical solution is computationally inexpensive.
-   **No Hyperparameters:** There are no parameters to tune, making it very simple to use.

#### Limitations
-   **Linearity Assumption:** Fails completely if the underlying relationship is non-linear.
-   **Sensitivity to Outliers:** Since errors are squared, a single outlier can heavily skew the line of best fit.
-   **Limited to One Feature:** By definition, it cannot handle multiple predictors.

#### Inductive Bias
The model's inductive bias is a strong assumption of **linearity**. It will always prefer a straight line to explain the data, regardless of the true underlying pattern.

#### Performance Analysis
```
Intercept: 6.1510
Slope: 0.7011
R-squared: 0.0021
```
-   **What the Metrics Say:** The **R-squared value of 0.0021 is extremely low**. This means the model explains only 0.21% of the variance in the target variable. This does not mean the model is broken; rather, it indicates that the synthetic data was generated with a very high amount of random noise, making the linear relationship between the feature and the target incredibly weak. The model has correctly identified the best possible straight line, but a straight line is a poor predictor for this noisy dataset.

#### Parameter Tuning
-   This specific implementation has **no hyperparameters to tune**. The OLS method provides a direct, deterministic solution. To "improve" the metrics, one would need to either gather less noisy data or choose a more complex model capable of fitting noise (which is generally undesirable).

---

### 2. Multiple & Polynomial Regression

#### Model Explanation
This file implements a **Multiple Linear Regression** model using **Gradient Descent**. It extends Simple Linear Regression to handle multiple input features ($X_1, X_2, ..., X_n$). The goal is to find a set of weights (coefficients) for each feature and a bias (intercept) that minimizes the Mean Squared Error (MSE). Gradient Descent is an iterative optimization algorithm that starts with random weights and repeatedly adjusts them in the direction that most steeply decreases the error, akin to walking downhill on the error surface.

**Polynomial Regression** is treated as a special case: we first create new features by raising the original feature to various powers (e.g., $X^2, X^3$). Then, we feed these transformed features into the Multiple Linear Regression model.

#### Advantages
-   **Handles Multiple Features:** Can model more complex, real-world scenarios.
-   **Captures Non-linearity (Polynomial):** Can fit curved data patterns effectively.
-   **Still Interpretable:** The magnitude of each weight indicates the importance of its corresponding feature.

#### Limitations
-   **Requires Feature Scaling:** Gradient Descent converges much faster and more reliably when features are on a similar scale.
-   **Prone to Overfitting (Polynomial):** A high-degree polynomial can perfectly fit the training data but will fail to generalize to new data.
-   **Assumes No Multicollinearity:** Performance degrades if features are highly correlated.

#### Inductive Bias
The inductive bias is **linearity in the parameters**. The model assumes that the target variable can be predicted as a weighted sum of the input features. For Polynomial Regression, the bias is that the relationship can be approximated by a polynomial function of a specific degree.

#### Performance Analysis
```
# Multiple Regression
R-squared: 0.3085

# Polynomial Regression (degree=2)
R-squared: 0.6265
```
-   **What the Metrics Say:** The Multiple Linear Regression model explains about 31% of the variance, which is a significant improvement over the simple model but still modest. However, the Polynomial model explains over 62% of the variance. This is a clear indicator that the underlying data for this problem had a quadratic (curved) relationship that the linear model couldn't capture but the polynomial model could.

#### Parameter Tuning
-   `learning_rate`: This is the most critical hyperparameter. It controls the step size of each gradient update.
    -   **Impact:** A large `learning_rate` can cause the optimizer to overshoot the minimum and fail to converge. A small `learning_rate` will converge very slowly and may get stuck in a local minimum.
-   `n_iterations`: The number of times the training loop runs.
    -   **Impact:** Too few iterations will result in an underfit model that hasn't converged. Too many can be computationally wasteful. One should train until the error stops decreasing significantly.
-   `degree` (for Polynomial Regression): The degree of the polynomial features to create.
    -   **Impact:** This directly controls the model's complexity and the Bias-Variance Tradeoff. A low degree (e.g., 1) has high bias and may underfit. A very high degree has high variance and will almost certainly overfit the training data.

---

### 3. Ridge and Lasso Regression

#### Model Explanation
Ridge and Lasso are regularized versions of linear regression, designed to combat overfitting and handle multicollinearity. They work by adding a penalty term to the cost function that discourages the model from assigning large weights to the features.
-   **Ridge Regression (L2 Regularization):** Adds a penalty proportional to the **sum of the squared weights**. This forces weights to be small but rarely exactly zero. It is excellent for reducing model complexity when many features are useful.
-   **Lasso Regression (L1 Regularization):** Adds a penalty proportional to the **sum of the absolute values of the weights**. This has a unique property: it can shrink some feature weights to be *exactly zero*, effectively performing automatic feature selection.

#### Advantages
-   **Reduces Overfitting:** The penalty term constrains the model, leading to better generalization.
-   **Handles Multicollinearity:** Regularization makes the model more stable when features are correlated.
-   **Feature Selection (Lasso):** Lasso provides a sparse model that is easier to interpret by eliminating irrelevant features.

#### Limitations
-   **Adds a Hyperparameter (`alpha`):** The strength of the penalty must be tuned.
-   **Feature Scaling is Crucial:** The penalty is applied to the weights, so if features have different scales, their weights will be penalized unfairly.

#### Inductive Bias
The inductive bias is a preference for **simpler models**. Ridge prefers models where the weights are small and distributed, while Lasso has a stronger bias for **sparse models** (where many weights are exactly zero).

#### Performance Analysis
```
# Ridge Regression
R-squared: 0.4879

# Lasso Regression
R-squared: 0.4874
Lasso Weights: [ 8.71  2.02  5.56  0.35  5.97 -1.16  0.91  1.43 -0.46 -0.26]
```
-   **What the Metrics Say:** Both models achieve an R-squared of around 49%, a significant improvement over the standard Multiple Linear Regression. This suggests that regularization was effective in creating a more robust model for this particular dataset. For the Lasso model, note that none of the weights are exactly zero. This implies that at the chosen `alpha`, the penalty was not strong enough to eliminate any features, or that all features had some predictive value.

#### Parameter Tuning
-   `alpha` (lambda): This parameter controls the strength of the regularization penalty.
    -   **Impact:** If `alpha = 0`, the models are identical to standard Linear Regression. As `alpha` increases, the penalty becomes stronger, and the model's weights are shrunk more aggressively towards zero. A very large `alpha` will lead to underfitting, as all weights will be close to zero. The optimal `alpha` is typically found using cross-validation.

---
---

## Model Deep Dive: Classification

### 4. Logistic Regression

#### Model Explanation
Despite its name, Logistic Regression is a fundamental algorithm for **binary classification**. It works by first calculating a weighted sum of the input features (just like linear regression). Instead of using this output directly, it passes the result through a **sigmoid (or logistic) function**. This function squishes any real-valued number into a range between 0 and 1, which can be interpreted as the probability of the sample belonging to the positive class (Class 1). A decision boundary (typically 0.5) is then used to classify the sample: if the probability is > 0.5, it's Class 1; otherwise, it's Class 0.

#### Advantages
-   **Interpretable:** The model's coefficients can be interpreted in terms of the odds of a sample belonging to a class.
-   **Fast and Efficient:** It is computationally inexpensive to train and predict.
-   **Good Baseline:** It provides a great starting point for any classification problem.

#### Limitations
-   **Linear Decision Boundary:** It can only learn linear separation lines (or hyperplanes). It will fail on problems where the classes are not linearly separable.
-   **Requires Preprocessing:** Like other gradient-based models, it benefits from feature scaling.

#### Inductive Bias
The inductive bias is a strong assumption that the data is **linearly separable**. The model will always prefer a single straight line (or hyperplane) to separate the classes.

#### Performance Analysis
```
Accuracy: 0.8450
Precision: 0.8350
Recall: 0.8600
F1 Score: 0.8473
Confusion Matrix: [[83 17], [14 86]]
```
-   **What the Metrics Say:** The model achieves a solid accuracy of 84.5%. The Precision and Recall are well-balanced, indicating the model is not biased towards one type of error. The confusion matrix shows that the model made 17 False Positive errors (predicting 1 when it was 0) and 14 False Negative errors (predicting 0 when it was 1). Overall, these results suggest that the synthetic data was mostly linearly separable and the model performed well.

#### Parameter Tuning
-   `learning_rate` & `n_iterations`: These function identically to their counterparts in linear regression, controlling the speed and duration of the Gradient Descent optimization. They must be tuned to ensure the model's cost function converges to its minimum.

---

### 5. Decision Tree Classifier

#### Model Explanation
A Decision Tree is a non-linear model that works by recursively partitioning the data into smaller and smaller subsets. It functions like a flowchart of "if-else" questions. At each step (a "node"), the algorithm selects the feature and the threshold that best splits the data. "Best" is defined as the split that results in the most separation between the classes in the resulting subsets (the "children nodes"). This is measured using **Information Gain**, which is based on **Entropy** (a measure of impurity or randomness). The process continues until a stopping criterion is met, at which point a final class label is assigned to the "leaf node".

#### Advantages
-   **Highly Interpretable:** The tree structure can be visualized and is easy for humans to understand.
-   **Handles Non-linear Data:** Can capture complex relationships without feature transformations.
-   **No Feature Scaling Required:** The splitting logic is not sensitive to the scale of the features.

#### Limitations
-   **Prone to Overfitting:** A deep tree can perfectly memorize the training data, including its noise, leading to poor performance on new data.
-   **Instability:** Small variations in the data can result in a completely different tree being generated.
-   **Biased Trees:** Can create biased trees if some classes are dominant.

#### Inductive Bias
The inductive bias is a preference for **hierarchical, axis-aligned splits**. The model prefers to create splits that are parallel to the feature axes. It also prefers simpler, shorter trees, although this preference must often be enforced through pruning.

#### Performance Analysis
```
Accuracy: 1.0000
Precision: 1.0000
Recall: 1.0000
F1 Score: 1.0000
```
-   **What the Metrics Say:** The model achieved perfect scores across the board. While this looks impressive, **this is a classic and dangerous sign of severe overfitting**. The Decision Tree was allowed to grow deep enough that it created a unique path for every single sample in the training data, effectively "memorizing" it. This model would likely perform very poorly on a separate test set because it has learned the noise, not the underlying pattern.

#### Parameter Tuning (Pruning)
To combat overfitting and create a model that generalizes, you must "prune" the tree by tuning its hyperparameters:
-   `max_depth`: This is the most important parameter. It limits the maximum number of splits from the root to any leaf.
    -   **Impact:** A smaller `max_depth` (e.g., 3, 5, 7) forces the model to learn more general patterns and drastically reduces overfitting.
-   `min_samples_split`: The minimum number of data points a node must contain to be considered for splitting.
    -   **Impact:** Setting this to a higher value (e.g., 5, 10) prevents the model from creating splits based on just a few samples, which are likely to be noise. This promotes a smoother, more generalized decision boundary.