# SUPERVISED LEARNING

Supervised Machine Learning is where models are trained on data to predict something. The data model is built in such a way that the predicted values and original values are as close/similar as possible.
There are two types of prediction models, where we predict either continuous outcome(Regression) or classifying an observation into a suitable category.(Classification).

Supervised learning is a machine learning approach where a model is trained on labeled data. Here's a list of commonly used models in supervised learning, categorized into different types:

---

### Contents

_**Basics**_

1. [Cost Fucntion and optimization](#computing-cost)
2. [Ordinary Least Squares](#ordinary-least-squares)
3. [Maximum Likelihood Estimation](#maximum-likelihood-estimation)
4. [Evaluation Metrics - Regression]()
5. [Regularization](#regularization)
6. [Gradient Descent](#gradient-descent)
7. [Stochastic Gradient Descent](#stochastic-gradient-descent)
8. [Bias Variance trade-off]()
9. [Regularization](#regularization)
10. [Data transformations]()
11. [Cross-Validation methods]()

_**Linear Models**_

1. Linear Regression
2. Logistic Regression (binary classification)
3. Ridge Regression (L2 regularization)
4. Lasso Regression (L1 regularization)
5. Elastic Net Regression (L1 + L2)
6. Generalized Linear Models(GLM)

_**Support Vector Machines (SVMs)**_

1. Linear SVM (for linearly separable data)
2. Kernel SVM (for non-linear data, using kernels like RBF, polynomial)

_**Decision Tree-Based Models**_

1. Decision Trees (for classification and regression)
2. Random Forest (ensemble of decision trees)
3. Gradient Boosting Machines (GBM):
   1. XGBoost
   2. LightGBM
   3. CatBoost
4. AdaBoost
5. Extra Trees (Extremely Randomized Trees)

_**Bayesian Models**_

1. Naive Bayes:
   1. Gaussian Naive Bayes
   2. Multinomial Naive Bayes
   3. Bernoulli Naive Bayes
2. Bayesian Linear Regression

_**Instance-Based Learning**_

1. k-Nearest Neighbors (kNN)

---

### BASICS

#### Computing Cost

In supervised learning, cost functions quantify the error between predicted and true values. The choice of a cost function depends on the **task type** (regression or classification), **model type**, and **data characteristics**.

$loss = e_{i} = \hat{y_{i}} - y_i$.

where $y_i$ is observed value of the dependent variable and $\hat{y_i}$ is the predicted value.

Cost is a measure of how well our model is predicting.
Loss is a how much error we get for a single observation, whereas the cost is the average of all the loss.

The equation for cost with one variable is:

$$
J(w,b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})^2
$$

where

$$
f_{w,b}(x^{(i)}) = wx^{(i)} + b
$$

- $f_{w,b}(x^{(i)})$ is our prediction for example $i$ using parameters $w,b$.
- $(f_{w,b}(x^{(i)}) - y^{(i)})^2$ is the squared difference between the target value and the prediction.
- These differences are summed over all the $m$ examples and divided by `2m` to produce the cost, $J(w,b)$.  
  The cost is a measure of how accurate the model is on the training data. The cost equation (1) above shows that if $w$ and $b$ can be selected such that the predictions $f_{w,b}(x)$ match the target data $y$, the $(f_{w,b}(x^{(i)}) - y^{(i)})^2$ term will be zero and the cost minimized.

##### Regression Cost Functions

These are used when the target variable is continuous.

###### (a) Mean Squared Error (MSE)

$$
J(\mathbf{w}) = \frac{1}{n} \sum\_{i=1}^n \left( y_i - \hat{y}\_i \right)^2
$$

- Penalizes large errors more heavily due to the square term.
- Commonly used in linear regression.

###### (b) Mean Absolute Error (MAE)

$$
J(\mathbf{w}) = \frac{1}{n} \sum\_{i=1}^n \left| y_i - \hat{y}\_i \right|
$$

- Less sensitive to outliers compared to MSE.
- Leads to a piecewise-linear optimization problem.

##### Classification Cost Functions

These are used when the target variable is categorical.

###### (a) Log Loss (Cross-Entropy Loss)

For binary classification:

$$
J(\mathbf{w}) = -\frac{1}{n} \sum_{i=1}^n \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
$$

For multi-class classification:

$$
J(\mathbf{w}) = -\frac{1}{n} \sum_{i=1}^n \sum_{k=1}^K y_{i,k} \log(\hat{y}_{i,k}),
$$

where $K$ is the number of classes.

- Measures the difference between true and predicted probability distributions.
- Commonly used in logistic regression and neural networks.

---

#### Ordinary Least Squares

To derive the formula for the **beta vector** ($\boldsymbol{\beta}$) in multiple linear regression, let's start from the basic principles. The model for multiple linear regression is:

$$
\mathbf{y} = \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\epsilon},
$$

where:

- $\mathbf{y}$ is the $n \times 1$ vector of observed dependent variables,
- $\mathbf{X}$ is the $n \times p$ matrix of predictors (independent variables),
- $\boldsymbol{\beta}$ is the $p \times 1$ vector of coefficients to be estimated,
- $\boldsymbol{\epsilon}$ is the $n \times 1$ vector of errors (assumed to be normally distributed with mean $0$ and variance $\sigma^2$).

###### Objective

We use the method of least squares, which minimizes the sum of squared errors:

$$
\text{Minimize } S(\boldsymbol{\beta}) = (\mathbf{y} - \mathbf{X} \boldsymbol{\beta})^\top (\mathbf{y} - \mathbf{X} \boldsymbol{\beta}).
$$

###### Derivation

1. Expand the objective function:

$$
S(\boldsymbol{\beta}) = (\mathbf{y} - \mathbf{X} \boldsymbol{\beta})^\top (\mathbf{y} - \mathbf{X} \boldsymbol{\beta}).
$$

Expanding the quadratic form:

$$
S(\boldsymbol{\beta}) = \mathbf{y}^\top \mathbf{y} - 2 \mathbf{y}^\top \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\beta}^\top \mathbf{X}^\top \mathbf{X} \boldsymbol{\beta}.
$$

2. Differentiate with respect to $\boldsymbol{\beta}$:

To find the minimum, take the derivative of $S(\boldsymbol{\beta})$ with respect to $\boldsymbol{\beta}$:

$$
\frac{\partial S(\boldsymbol{\beta})}{\partial \boldsymbol{\beta}} = -2 \mathbf{X}^\top \mathbf{y} + 2 \mathbf{X}^\top \mathbf{X} \boldsymbol{\beta}.
$$

3. Set the derivative to zero:

$$
-2 \mathbf{X}^\top \mathbf{y} + 2 \mathbf{X}^\top \mathbf{X} \boldsymbol{\beta} = 0.
$$

Simplify:

$$
\mathbf{X}^\top \mathbf{y} = \mathbf{X}^\top \mathbf{X} \boldsymbol{\beta}.
$$

4. Solve for $\boldsymbol{\beta}$:

Assuming $\mathbf{X}^\top \mathbf{X}$ is invertible (non-singular):

$$
\boldsymbol{\beta} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}.
$$

###### Final Formula

The estimated coefficients in multiple linear regression are given by:

$$
\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}.
$$

###### Conditions

For this derivation to hold:

1. $\mathbf{X}^\top \mathbf{X}$ must be invertible (i.e., $\mathbf{X}$ has full column rank).
2. The errors $\boldsymbol{\epsilon}$ should ideally be homoscedastic and normally distributed for valid inference.

###### Residual Sum of Squares (RSS)

The Residual Sum of Squares (RSS) is:

$$
\text{RSS} = \|\mathbf{r}\|^2 = (\mathbf{y} - \mathbf{X} \hat{\boldsymbol{\beta}})^\top (\mathbf{y} - \mathbf{X} \hat{\boldsymbol{\beta}}).
$$

###### Variance Estimate

The unbiased estimate of the variance $\sigma^2$ is:

$$
\hat{\sigma}^2 = \frac{\text{RSS}}{n - p},
$$

where:

- $n$ is the number of observations,
- $p$ is the number of parameters (including the intercept).

This adjustment by dividing by $n - p$ instead of $n$ accounts for the degrees of freedom used in estimating $\boldsymbol{\beta}$.

In matrix terms:

$$
\hat{\sigma}^2 = \frac{(\mathbf{y} - \mathbf{X} \hat{\boldsymbol{\beta}})^\top (\mathbf{y} - \mathbf{X} \hat{\boldsymbol{\beta}})}{n - p}.
$$

###### Derivation of Degrees of Freedom

1. The residuals $\mathbf{r}$ are restricted by the model fit because $\mathbf{X} \hat{\boldsymbol{\beta}}$ lies in the column space of $\mathbf{X}$.
2. The rank of $\mathbf{X}$ is $p$, so $n - p$ is the number of free residuals or the degrees of freedom for error.

###### Properties of the Variance Estimate

1. **Unbiasedness**: $\mathbb{E}[\hat{\sigma}^2] = \sigma^2$.
2. **Interpretation**: $\hat{\sigma}^2$ represents the estimated variance of the noise in the data.

###### Connection to MLE

The maximum likelihood estimate (MLE) of $\sigma^2$ is:

$$
\hat{\sigma}^2\_{\text{MLE}} = \frac{\text{RSS}}{n}.
$$

Note: This differs from the OLS variance estimate, which divides by $n - p$, making OLS unbiased.

---

#### Maximum Likelihood Estimation (MLE)

##### The Model

The multiple linear regression model is given by:

$$
\mathbf{y} = \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\epsilon},
$$

where:

- $\mathbf{y}$ is the $n \times 1$ vector of observed dependent variables,
- $\mathbf{X}$ is the $n \times p$ matrix of predictors (including an intercept column if needed),
- $\boldsymbol{\beta}$ is the $p \times 1$ vector of coefficients,
- $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$ is the error vector, assumed to follow a multivariate normal distribution with mean 0 and covariance matrix $\sigma^2 \mathbf{I}$.

##### Likelihood Function

The probability density function of $\mathbf{y}$, given $\mathbf{X}$ and $\boldsymbol{\beta}$, is:

$$
\mathbf{y} \sim \mathcal{N}(\mathbf{X} \boldsymbol{\beta}, \sigma^2 \mathbf{I}),
$$

with the likelihood:

$$
L(\boldsymbol{\beta}, \sigma^2 \mid \mathbf{y}) = \frac{1}{(2 \pi \sigma^2)^{n/2}} \exp\left(-\frac{1}{2 \sigma^2} \| \mathbf{y} - \mathbf{X} \boldsymbol{\beta} \|^2 \right).
$$

##### Log-Likelihood Function

Taking the natural logarithm of the likelihood simplifies the expression:

$$
\ell(\boldsymbol{\beta}, \sigma^2 \mid \mathbf{y}) = -\frac{n}{2} \log(2 \pi) - \frac{n}{2} \log(\sigma^2) - \frac{1}{2 \sigma^2} \| \mathbf{y} - \mathbf{X} \boldsymbol{\beta} \|^2.
$$

##### Maximizing the Log-Likelihood

**Step 1: Maximize with Respect to $\boldsymbol{\beta}$**

The first term, $-\frac{1}{2 \sigma^2} \| \mathbf{y} - \mathbf{X} \boldsymbol{\beta} \|^2$, depends on $\boldsymbol{\beta}$. To maximize this, minimize $\| \mathbf{y} - \mathbf{X} \boldsymbol{\beta} \|^2$. The solution is the ordinary least squares (OLS) estimate:

$$
\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}.
$$

**Step 2: Maximize with Respect to $\sigma^2$**

Substitute $\hat{\boldsymbol{\beta}}$ into the log-likelihood function. The maximum likelihood estimate for $\sigma^2$ is derived by maximizing:

$$
\ell(\sigma^2 \mid \mathbf{y}, \hat{\boldsymbol{\beta}}) = -\frac{n}{2} \log(2 \pi) - \frac{n}{2} \log(\sigma^2) - \frac{1}{2 \sigma^2} \| \mathbf{y} - \mathbf{X} \hat{\boldsymbol{\beta}} \|^2.
$$

Taking the derivative with respect to $\sigma^2$ and setting it to zero gives:

$$
\hat{\sigma}^2 = \frac{\| \mathbf{y} - \mathbf{X} \hat{\boldsymbol{\beta}} \|^2}{n}.
$$

##### Final MLE Estimates

The maximum likelihood estimates are:

1. Coefficients:

   $$
   \hat{\boldsymbol{\beta}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}.
   $$

2. Variance of errors:

   $$
   \hat{\sigma}^2 = \frac{\| \mathbf{y} - \mathbf{X} \hat{\boldsymbol{\beta}} \|^2}{n}.
   $$

##### Connection to OLS

- The MLE for $\boldsymbol{\beta}$ in linear regression is the same as the OLS estimator.
- The MLE for $\sigma^2$ is slightly different from the unbiased sample variance since it divides by $n$, not $n - p$.

---

#### Gradient Descent

_Gradient Descent_ is an optimization algorithm used to minimize a cost function (or loss function) by iteratively adjusting the model parameters in the direction of the steepest descent of the function.

##### Single Variable

The equation for cost with one variable is:

$$
J(w,b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})^2
$$

where

$$
f_{w,b}(x^{(i)}) = wx^{(i)} + b
$$

- $f_{w,b}(x^{(i)})$ is our prediction for example $i$ using parameters $w,b$.
- $(f_{w,b}(x^{(i)}) - y^{(i)})^2$ is the squared difference between the target value and the prediction.
- These differences are summed over all the $m$ examples and divided by `2m` to produce the cost, $J(w,b)$.  
  The cost is a measure of how accurate the model is on the training data. The cost equation (1) above shows that if $w$ and $b$ can be selected such that the predictions $f_{w,b}(x)$ match the target data $y$, the $(f_{w,b}(x^{(i)}) - y^{(i)})^2$ term will be zero and the cost minimized.

repeat until convergence:

$$
w = w -  \alpha \frac{\partial J(w,b)}{\partial w}
$$

$$
b = b -  \alpha \frac{\partial J(w,b)}{\partial b}
$$

where, parameters $w$, $b$ are updated simultaneously.  
The gradient is defined as:

$$
\frac{\partial J(w,b)}{\partial w}  = \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})x^{(i)}
$$

$$
\frac{\partial J(w,b)}{\partial b}  = \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})
$$

Here _simultaniously_ means that you calculate the partial derivatives for all the parameters before updating any of the parameters.

##### Multiple Variables

$$
f_{\mathbf{w}, b}(X) = b + w_1 x_{i1} + w_2 x_{i2} + \dots + w_p x_{ip} \newline
$$

$$
J(\mathbf{w}, b) = \frac{1}{n} \left( \mathbf{y} - \mathbf{X} \mathbf{w} \right)^\top \left( \mathbf{y} - \mathbf{X} \mathbf{w} \right)
$$

$X$ is a matrix with first column with all values of 1 to calculate the intercept. <br>
repeat until convergence:<br>

$$
w_j = w_j -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial w_j} \text{for j = 0 ... (n-1)} \\
$$

$$
b = b -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial b}
$$

where, n is the number of features, parameters $w_j$, $b$, are updated simultaneously and where

$$
\frac{\partial J(\mathbf{w},b)}{\partial w_j}  = \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})x_{j}^{(i)}
$$

$$
\frac{\partial J(\mathbf{w},b)}{\partial b}  = \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})
$$

- m is the number of training examples in the data set

- $f_{\mathbf{w},b}(\mathbf{x}^{(i)})$ is the model's prediction, while $y^{(i)}$ is the target value

---

#### Stochastic Gradient Descent

**Stochastic Gradient Descent (SGD)** is an optimization algorithm used to minimize a cost function by iteratively updating the model parameters in the direction of the negative gradient of the cost function. Unlike batch gradient descent, which computes the gradient using the entire dataset, SGD updates parameters for each training sample, making it faster and more suitable for large datasets.

1. **Gradient Descent**:

   - Gradient Descent optimizes a function by moving iteratively in the opposite direction of its gradient.
   - For a cost function $J(\boldsymbol{\theta})$, the update rule is:

     $$
     \boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k - \eta \nabla J(\boldsymbol{\theta}_k),
     $$

     where:

     - $\boldsymbol{\theta}_k$: Current parameter values,
     - $\eta$: Learning rate,
     - $\nabla J(\boldsymbol{\theta}_k)$: Gradient of the cost function.

2. **Stochastic Gradient Descent**:

   - Instead of using the full dataset to compute the gradient, SGD uses a single training sample (or a small batch) at each step.
   - For a dataset with $n$ samples $\{ (\mathbf{x}_i, y_i) \}$, the update rule is:

     $$
     \boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k - \eta \nabla J_i(\boldsymbol{\theta}_k),
     $$

     where $J_i(\boldsymbol{\theta}_k)$ is the cost function for the $i$-th sample.

<u>_**Steps in Stochastic Gradient Descent**_</u>

1. _Initialization_:

   - Start with initial values for the parameters $\boldsymbol{\theta}_0$.
   - Choose a learning rate $\eta$.

2. _Shuffle the Dataset_:

   - Randomly shuffle the dataset to prevent patterns in the data from affecting the optimization process.

3. _Iterative Updates_:

   - For each epoch (complete pass through the dataset):

     - For each sample $(\mathbf{x}_i, y_i)$:

       - Compute the gradient $\nabla J_i(\boldsymbol{\theta})$.
       - Update the parameters:

         $$
         \boldsymbol{\theta} = \boldsymbol{\theta} - \eta \nabla J_i(\boldsymbol{\theta}).
         $$

4. _Stopping Criteria_:
   - Stop when the parameters converge (small change in $\boldsymbol{\theta}$) or after a fixed number of epochs.

---

#### Evaluation Metrics - Regression

##### Mean Squared Error (MSE) 🏹

**Definition**: MSE measures the average of the squares of the errors—that is, the average squared difference between the observed actual outcomes and the predicted outcomes.

**Formula**:
$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

where:

- $n$ is the number of observations
- $y_i$ is the actual value
- $\hat{y}_i$ is the predicted value

##### Root Mean Squared Error (RMSE) 👑

**Definition**: RMSE is the square root of MSE and provides a measure of how well the model predictions match the observed data. RMSE is in the same units as the target variable, making it more interpretable.

**Formula**:
$$RMSE = \sqrt{MSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

##### R-squared (Coefficient of Determination) 🐇

**Definition**: R-squared measures the proportion of the variance in the dependent variable that is predictable from the independent variables. It provides an indication of the goodness of fit of a model.

**Formula**:
$$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

where:

- $\bar{y}$ is the mean of the actual values

##### Adjusted R-squared 🐼

**Definition**: Adjusted R-squared adjusts the R-squared value for the number of predictors in the model. It accounts for the fact that adding more predictors to a model will almost always increase the R-squared value, regardless of the actual improvement in model fit.

**Formula**:
$$\text{Adjusted } R^2 = 1 - \left( \frac{(1 - R^2)(n - 1)}{n - p - 1} \right)$$

where:

- $n$ is the number of observations
- $p$ is the number of predictors

##### Example

Let's consider the following example dataset with actual and predicted values:

| Observation | Actual ($y_i$) | Predicted ($\hat{y}_i$) |
| ----------- | -------------- | ----------------------- |
| 1           | 3              | 2.5                     |
| 2           | 4              | 3.8                     |
| 3           | 5              | 5.2                     |
| 4           | 6              | 6.1                     |
| 5           | 7              | 6.9                     |

**Mean Squared Error (MSE)**

First, calculate the squared errors:

$$
\begin{align*}
(3 - 2.5)^2 &= 0.25 \\
(4 - 3.8)^2 &= 0.04 \\
(5 - 5.2)^2 &= 0.04 \\
(6 - 6.1)^2 &= 0.01 \\
(7 - 6.9)^2 &= 0.01 \\
\end{align*}
$$

Then, compute the MSE:

$$
MSE = \frac{0.25 + 0.04 + 0.04 + 0.01 + 0.01}{5} = \frac{0.35}{5} = 0.07
$$

**Root Mean Squared Error (RMSE)**

Calculate the RMSE:

$$
RMSE = \sqrt{0.07} \approx 0.2646
$$

**R-squared (Coefficient of Determination)**

Calculate the total sum of squares (TSS):

$$
TSS = \sum_{i=1}^{5} (y_i - \bar{y})^2
$$

where $\bar{y}$ is the mean of actual values:

$$
\bar{y} = \frac{3 + 4 + 5 + 6 + 7}{5} = 5
$$

$$
TSS = (3 - 5)^2 + (4 - 5)^2 + (5 - 5)^2 + (6 - 5)^2 + (7 - 5)^2 = 4 + 1 + 0 + 1 + 4 = 10
$$

Now, calculate the explained sum of squares (ESS):

$$
ESS = \sum_{i=1}^{5} (y_i - \hat{y}_i)^2 = 0.35
$$

Then, compute R-squared:

$$
R^2 = 1 - \frac{0.35}{10} = 1 - 0.035 = 0.965
$$

**Adjusted R-squared**

Finally, calculate the Adjusted R-squared:

$$
\text{Adjusted } R^2 = 1 - \left( \frac{(1 - 0.965)(5 - 1)}{5 - 1 - 1} \right) = 1 - \left( \frac{(0.035)(4)}{3} \right) = 1 - 0.0467 \approx 0.9533
$$

---

#### Evaluation Metrics - Classification

These metrics are commonly used in evaluating the performance of classification models. They are derived from a confusion matrix, which summarizes the predictions of a binary classifier.

##### Confusion Matrix

For a binary classification problem:
| | Predicted Positive | Predicted Negative |
|----------------|---------------------|---------------------|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

- **True Positive (TP):** Correctly predicted positive cases.
- **True Negative (TN):** Correctly predicted negative cases.
- **False Positive (FP):** Negative cases incorrectly predicted as positive.
- **False Negative (FN):** Positive cases incorrectly predicted as negative.

---

##### Accuracy 🎯

Accuracy measures the proportion of correctly classified instances out of the total instances.

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

##### Precision (Positive Predictive Value) 🧮

Precision measures the proportion of true positive predictions out of all positive predictions.

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

##### Recall (Sensitivity) 🩺

Recall measures the proportion of actual positives correctly identified.

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

##### F1 Score ⚖️

The F1 Score is the harmonic mean of precision and recall, balancing the two.

$$
\text{F1 Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

##### Specificity (True Negative Rate) ✅

Specificity measures the proportion of actual negatives correctly identified.

$$
\text{Specificity} = \frac{TN}{TN + FP}
$$

##### Area Under the ROC Curve (AUC-ROC) 📈

The AUC-ROC measures the ability of a classifier to distinguish between classes across all thresholds. The Receiver Operating Characteristic (ROC) curve plots **True Positive Rate (TPR)** vs. **False Positive Rate (FPR)** for various thresholds.

- **TPR (Recall/Sensitivity):**
  $$
  TPR = \frac{TP}{TP + FN}
  $$
- **FPR:**
  $$
  FPR = \frac{FP}{FP + TN}
  $$
  The AUC is the integral of the ROC curve and represents the probability that a randomly chosen positive instance is ranked higher than a randomly chosen negative instance.

$$
\text{AUC} = \int_0^1 TPR(FPR) \, d(FPR)
$$

##### Matthews Correlation Coefficient (MCC) 🔗

The MCC is a balanced metric that considers all elements of the confusion matrix, even in imbalanced datasets. It ranges from -1 to +1:

- +1: Perfect prediction.
- 0: Random guessing.
- -1: Total disagreement.

$$
\text{MCC} = \frac{(TP \cdot TN) - (FP \cdot FN)}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}
$$

##### Logarithmic Loss (Log Loss) 📉

Log Loss measures the uncertainty of predictions by penalizing incorrect confidence levels. Lower Log Loss indicates better predictions.

$$
\text{Log Loss} = -\frac{1}{N} \sum_{i=1}^{N} \left( y_i \cdot \log(p_i) + (1 - y_i) \cdot \log(1 - p_i) \right)
$$

- $y_i$: Actual label (0 or 1).
- $p_i$: Predicted probability of the positive class.
- $N$: Total number of samples.

##### Cohen's Kappa ⚖️

Cohen's Kappa measures inter-rater agreement while accounting for the probability of agreement by chance. It ranges from -1 (total disagreement) to 1 (perfect agreement).

$$
\text{Kappa} = \frac{P_o - P_e}{1 - P_e}
$$

- $P_o$: Observed agreement ($\frac{TP + TN}{N}$).
- $P_e$: Expected agreement based on random chance.

$$
P_e = \frac{(TP + FP)(TP + FN) + (TN + FP)(TN + FN)}{N^2}
$$

---

##### Example

###### Dataset

**Actual Values:** `[1, 0, 1, 1, 0, 0, 1, 0, 0, 1]`  
**Predicted Probabilities:** `[0.9, 0.3, 0.8, 0.4, 0.2, 0.1, 0.7, 0.6, 0.2, 0.9]`  
**Predicted Labels (Threshold = 0.5):** `[1, 0, 1, 0, 0, 0, 1, 1, 0, 1]`

###### Confusion Matrix

|                 | Predicted Positive | Predicted Negative |
| --------------- | ------------------ | ------------------ |
| Actual Positive | 459 (TP)           | 51 (FN)            |
| Actual Negative | 90 (FP)            | 400 (TN)           |

###### Metrics Calculation

1. Accuracy

   $$
   \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{459 + 400}{459 + 400 + 90 + 51} = \frac{859}{1000} = 0.859
   $$

2. Precision

   $$
   \text{Precision} = \frac{TP}{TP + FP} = \frac{459}{459 + 90} = \frac{459}{549} \approx 0.836
   $$

3. Recall (Sensitivity)

   $$
   \text{Recall} = \frac{TP}{TP + FN} = \frac{459}{459 + 51} = \frac{459}{510} \approx 0.900
   $$

4. F1 Score

   $$
   F1\ \text{Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = 2 \cdot \frac{0.836 \cdot 0.900}{0.836 + 0.900} \approx 0.867
   $$

5. Specificity

   $$
   \text{Specificity} = \frac{TN}{TN + FP} = \frac{400}{400 + 90} = \frac{400}{490} \approx 0.816
   $$

6. Area Under the ROC Curve (AUC-ROC)

   For AUC-ROC, we need the True Positive Rate (TPR) and False Positive Rate (FPR):

   $$
   TPR = \text{Recall} = 0.900
   $$

   $$
   FPR = \frac{FP}{FP + TN} = \frac{90}{90 + 400} = \frac{90}{490} \approx 0.184
   $$

   The AUC-ROC can be calculated using these rates, typically through integration or numerical methods. For simplicity, we assume an approximate AUC-ROC value based on the TPR and FPR, such as around 0.858.

7. Matthews Correlation Coefficient (MCC)

   $$
   \text{MCC} = \frac{(TP \cdot TN) - (FP \cdot FN)}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}} \\ = \frac{(459 \cdot 400) - (90 \cdot 51)}{\sqrt{(459 + 90)(459 + 51)(400 + 90)(400 + 51)}} \approx 0.718
   $$

8. Logarithmic Loss (Log Loss)

   We need the predicted probabilities for Log Loss. Assume we have predicted probabilities for the positive class; let's consider it for 10 instances (just an example):

   - Actual: [1, 1, 1, 0, 0, 1, 0, 0, 1, 0]
   - Predicted Probability: [0.9, 0.8, 0.7, 0.4, 0.3, 0.6, 0.2, 0.1, 0.9, 0.05]
     $$
     \text{Log Loss} = -\frac{1}{10} \sum_{i=1}^{10} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right] \approx 0.216
     $$

9. Cohen's Kappa

   $$
   p_o = \frac{(TP + TN)}{1000} = \frac{(459 + 400)}{1000} = 0.859
   $$

   $$
   p_e = \left(\frac{(459 + 51)}{1000} \times \frac{(459 + 90)}{1000}\right) + \left(\frac{(400 + 90)}{1000} \times \frac{(400 + 51)}{1000}\right) \approx 0.509
   $$

   $$
   \kappa = \frac{0.859 - 0.509}{1 - 0.509} \approx 0.714
   $$

###### Summary Table

| Metric           | Value |
| ---------------- | ----- |
| Accuracy         | 0.859 |
| Precision        | 0.836 |
| Recall           | 0.900 |
| F1 Score         | 0.867 |
| Specificity      | 0.816 |
| AUC-ROC          | 0.858 |
| Confusion Matrix | Shown |
| MCC              | 0.718 |
| Log Loss         | 0.216 |
| Cohen's Kappa    | 0.509 |

---

- **High precision** ensures fewer false positives.
- **High recall** ensures fewer false negatives.
- **F1 Score** balances precision and recall.
- **Specificity** complements recall by focusing on true negatives.
- **Accuracy** provides an overall performance measure but may be misleading if the dataset is imbalanced.
- **AUC-ROC** highlights the model's ability to rank predictions, especially useful for imbalanced datasets.
- **Confusion Matrix** provides a breakdown of true and false predictions.
- **MCC** is a robust metric for balanced evaluation, even in imbalanced datasets.
- **Log Loss** evaluates probabilistic predictions and penalizes overconfidence in incorrect predictions.
- **Cohen's Kappa** adjusts accuracy for chance agreement, making it valuable for agreement analysis.

---

#### Regularization

**Regularization** is a technique used to prevent overfitting by adding a penalty term to the cost function. It discourages the model from fitting too closely to the training data, which improves generalization to unseen data.

1. <u>_Overfitting_</u>: Complex models with many parameters (e.g., high-dimensional regression) can overfit the training data.
2. <u>_Multicollinearity_</u>: In regression, highly correlated predictors can make coefficient estimates unstable.
3. <u>_Improved Generalization_</u>: Regularization helps simplify the model by shrinking parameters.

##### Regularized Cost Function

For a regression or classification model, the general cost function becomes:

$$
J(\boldsymbol{\beta}) = \text{Loss Function}(\boldsymbol{\beta}) + \lambda \cdot \text{Regularization Term},
$$

where:

- **Loss Function**: Measures the prediction error (e.g., Mean Squared Error for regression, Cross-Entropy for classification).
- **Regularization Term**: Penalizes large or complex parameter values (e.g., $\|\boldsymbol{\beta}\|_1$ or $\|\boldsymbol{\beta}\|_2$).
- $\lambda$: Regularization strength (hyperparameter).

##### Types of Regularization

1. **L1 Regularization (Lasso)**:

   $$
   J(\boldsymbol{\beta}) = \text{Loss Function}(\boldsymbol{\beta}) + \lambda \|\boldsymbol{\beta}\|_1,
   $$

   where $\|\boldsymbol{\beta}\|\_1 = \sum_{j=1}^p |\beta_j|$.

   - Encourages sparsity by shrinking some coefficients to zero.
   - Used for feature selection.

2. **L2 Regularization (Ridge)**:

   $$
   J(\boldsymbol{\beta}) = \text{Loss Function}(\boldsymbol{\beta}) + \lambda \|\boldsymbol{\beta}\|_2^2,
   $$

   where $\|\boldsymbol{\beta}\|\_2^2 = \sum_{j=1}^p \beta_j^2$.

   - Penalizes large coefficients but doesn’t force them to zero.
   - Stabilizes models in the presence of multicollinearity.

3. **Elastic Net**:

   $$
   J(\boldsymbol{\beta}) = \text{Loss Function}(\boldsymbol{\beta}) + \lambda_1 \|\boldsymbol{\beta}\|_1 + \lambda_2 \|\boldsymbol{\beta}\|_2^2.
   $$

   - Combines L1 and L2 regularization for better flexibility.

##### Estimating Parameters

###### 1. Regression Models

The general regression model is:

$$
\mathbf{y} = \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\epsilon}.
$$

The regularized cost function is:

$$
J(\boldsymbol{\beta}) = \|\mathbf{y} - \mathbf{X} \boldsymbol{\beta}\|^2 + \lambda R(\boldsymbol{\beta}),
$$

where $R(\boldsymbol{\beta})$ is the regularization term.

- **Ridge Regression (L2 Regularization)**:

  $$
  \hat{\boldsymbol{\beta}} = (\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^\top \mathbf{y}.
  $$

- **Lasso Regression (L1 Regularization)**:
  The optimization involves minimizing an objective with an L1 penalty. This is solved using techniques like:

  - Coordinate Descent,
  - Least Angle Regression (LARS).

- **Elastic Net**:
  Solved using numerical methods such as gradient descent or coordinate descent.

###### 2. Classification Models

For classification, regularization is added to the cost function, often based on **Cross-Entropy Loss**:

$$
J(\boldsymbol{\beta}) = -\frac{1}{n} \sum_{i=1}^n \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right] + \lambda R(\boldsymbol{\beta}).
$$

- **Logistic Regression with L2 Regularization**:
  Solved using:

  - Gradient Descent,
  - Newton’s Method.

- **Neural Networks**:
  Regularization terms are added to the total loss function during training (e.g., L2 weight decay).

##### Hyperparameter Tuning

1. **Regularization Strength ($\lambda$)**:

   - Small $\lambda$: Weak regularization, model may overfit.
   - Large $\lambda$: Strong regularization, model may underfit.
   - Use cross-validation to select $\lambda$.

2. **Elastic Net Parameters**:
   - $\alpha$: Determines the mix between L1 and L2 regularization.

##### Summary

1. Regularization penalizes large or complex model parameters to improve generalization.
2. Parameter estimation involves solving the modified cost function using analytical or numerical techniques.
3. Cross-validation is essential for selecting the optimal regularization strength.

### Linear Models

#### Linear Regression

#### Logistic Regression

#### Ridge Regression
