# SUPERVISED LEARNING

Supervised Machine Learning is where models are trained on data to predict something. The data model is built in such a way that the predicted values and original values are as close/similar as possible.
There are two types of prediction models, where we predict either continuous outcome(Regression) or classifying an observation into a suitable category.(Classification).

Supervised learning is a machine learning approach where a model is trained on labeled data. Here's a list of commonly used models in supervised learning, categorized into different types:

---

## Contents

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

1. [Linear Regression](#linear-regression)

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

## BASICS

### Computing Cost

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

#### Regression Cost Functions

These are used when the target variable is continuous.

##### (a) Mean Squared Error (MSE)

$$
J(\mathbf{w}) = \frac{1}{n} \sum\_{i=1}^n \left( y_i - \hat{y}\_i \right)^2
$$

- Penalizes large errors more heavily due to the square term.
- Commonly used in linear regression.

##### (b) Mean Absolute Error (MAE)

$$
J(\mathbf{w}) = \frac{1}{n} \sum\_{i=1}^n \left| y_i - \hat{y}\_i \right|
$$

- Less sensitive to outliers compared to MSE.
- Leads to a piecewise-linear optimization problem.

#### Classification Cost Functions

These are used when the target variable is categorical.

##### (a) Log Loss (Cross-Entropy Loss)

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

### Ordinary Least Squares

To derive the formula for the **beta vector** ($\boldsymbol{\beta}$) in multiple linear regression, let's start from the basic principles. The model for multiple linear regression is:

$$
\mathbf{y} = \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\epsilon},
$$

where:

- $\mathbf{y}$ is the $n \times 1$ vector of observed dependent variables,
- $\mathbf{X}$ is the $n \times p$ matrix of predictors (independent variables),
- $\boldsymbol{\beta}$ is the $p \times 1$ vector of coefficients to be estimated,
- $\boldsymbol{\epsilon}$ is the $n \times 1$ vector of errors (assumed to be normally distributed with mean $0$ and variance $\sigma^2$).

#### Objective

We use the method of least squares, which minimizes the sum of squared errors:

$$
\text{Minimize } S(\boldsymbol{\beta}) = (\mathbf{y} - \mathbf{X} \boldsymbol{\beta})^\top (\mathbf{y} - \mathbf{X} \boldsymbol{\beta}).
$$

#### Derivation

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

#### Final Formula

The estimated coefficients in multiple linear regression are given by:

$$
\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}.
$$

#### Conditions

For this derivation to hold:

1. $\mathbf{X}^\top \mathbf{X}$ must be invertible (i.e., $\mathbf{X}$ has full column rank).
2. The errors $\boldsymbol{\epsilon}$ should ideally be homoscedastic and normally distributed for valid inference.

#### Residual Sum of Squares (RSS)

The Residual Sum of Squares (RSS) is:

$$
\text{RSS} = \|\mathbf{r}\|^2 = (\mathbf{y} - \mathbf{X} \hat{\boldsymbol{\beta}})^\top (\mathbf{y} - \mathbf{X} \hat{\boldsymbol{\beta}}).
$$

#### Variance Estimate

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

#### Derivation of Degrees of Freedom

1. The residuals $\mathbf{r}$ are restricted by the model fit because $\mathbf{X} \hat{\boldsymbol{\beta}}$ lies in the column space of $\mathbf{X}$.
2. The rank of $\mathbf{X}$ is $p$, so $n - p$ is the number of free residuals or the degrees of freedom for error.

#### Properties of the Variance Estimate

1. **Unbiasedness**: $\mathbb{E}[\hat{\sigma}^2] = \sigma^2$.
2. **Interpretation**: $\hat{\sigma}^2$ represents the estimated variance of the noise in the data.

#### Connection to MLE

The maximum likelihood estimate (MLE) of $\sigma^2$ is:

$$
\hat{\sigma}^2\_{\text{MLE}} = \frac{\text{RSS}}{n}.
$$

Note: This differs from the OLS variance estimate, which divides by $n - p$, making OLS unbiased.

---

### Maximum Likelihood Estimation (MLE)

#### The Model

The multiple linear regression model is given by:

$$
\mathbf{y} = \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\epsilon},
$$

where:

- $\mathbf{y}$ is the $n \times 1$ vector of observed dependent variables,
- $\mathbf{X}$ is the $n \times p$ matrix of predictors (including an intercept column if needed),
- $\boldsymbol{\beta}$ is the $p \times 1$ vector of coefficients,
- $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$ is the error vector, assumed to follow a multivariate normal distribution with mean 0 and covariance matrix $\sigma^2 \mathbf{I}$.

#### Likelihood Function

The probability density function of $\mathbf{y}$, given $\mathbf{X}$ and $\boldsymbol{\beta}$, is:

$$
\mathbf{y} \sim \mathcal{N}(\mathbf{X} \boldsymbol{\beta}, \sigma^2 \mathbf{I}),
$$

with the likelihood:

$$
L(\boldsymbol{\beta}, \sigma^2 \mid \mathbf{y}) = \frac{1}{(2 \pi \sigma^2)^{n/2}} \exp\left(-\frac{1}{2 \sigma^2} \| \mathbf{y} - \mathbf{X} \boldsymbol{\beta} \|^2 \right).
$$

#### Log-Likelihood Function

Taking the natural logarithm of the likelihood simplifies the expression:

$$
\ell(\boldsymbol{\beta}, \sigma^2 \mid \mathbf{y}) = -\frac{n}{2} \log(2 \pi) - \frac{n}{2} \log(\sigma^2) - \frac{1}{2 \sigma^2} \| \mathbf{y} - \mathbf{X} \boldsymbol{\beta} \|^2.
$$

#### Maximizing the Log-Likelihood

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

#### Final MLE Estimates

The maximum likelihood estimates are:

1. Coefficients:

   $$
   \hat{\boldsymbol{\beta}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}.
   $$

2. Variance of errors:

   $$
   \hat{\sigma}^2 = \frac{\| \mathbf{y} - \mathbf{X} \hat{\boldsymbol{\beta}} \|^2}{n}.
   $$

#### Connection to OLS

- The MLE for $\boldsymbol{\beta}$ in linear regression is the same as the OLS estimator.
- The MLE for $\sigma^2$ is slightly different from the unbiased sample variance since it divides by $n$, not $n - p$.

---

### Gradient Descent

_Gradient Descent_ is an optimization algorithm used to minimize a cost function (or loss function) by iteratively adjusting the model parameters in the direction of the steepest descent of the function.

#### Single Variable

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

#### Multiple Variables

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

### Stochastic Gradient Descent

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

### Learning Rate

The learning rate is a critical hyperparameter in the gradient descent optimization algorithm, as it determines the size of the steps taken to reach the minimum of the cost function. Let's explore the effects of having a learning rate that is too low or too high:

#### Learning Rate Too Low

When the learning rate is too low, the algorithm takes very small steps toward the minimum. This can have several consequences:

- **Slow Convergence**: The algorithm may take a very long time to converge to the minimum, requiring a large number of iterations.
- **Risk of Getting Stuck**: It might get stuck in local minima or saddle points for extended periods.
- **Inefficiency**: Computational resources are used inefficiently, and it may seem like the algorithm is not making progress.

#### Learning Rate Too High

When the learning rate is too high, the algorithm takes large steps toward the minimum. This can lead to several issues:

- **Overshooting**: The algorithm may overshoot the minimum, causing it to diverge rather than converge. It jumps back and forth around the minimum without settling down.
- **Instability**: High learning rates can cause the cost function to fluctuate wildly, making it difficult for the algorithm to find a stable solution.
- **Non-Convergence**: In extreme cases, the algorithm may fail to converge altogether and will not find the minimum.

#### Finding the Right Learning Rate

To find an appropriate learning rate, you can:

- **Experiment**: Start with a small learning rate and gradually increase it while monitoring the cost function.
- **Learning Rate Schedules**: Use techniques like learning rate decay, where the learning rate decreases over time.
- **Adaptive Methods**: Use optimization algorithms like Adam, Adagrad, or RMSprop that adjust the learning rate dynamically.

Finding the right balance is crucial for the efficiency and effectiveness of the gradient descent algorithm.

---

### Evaluation Metrics - Regression

#### Mean Squared Error (MSE) 🏹

**Definition**: MSE measures the average of the squares of the errors—that is, the average squared difference between the observed actual outcomes and the predicted outcomes.

**Formula**:
$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

where:

- $n$ is the number of observations
- $y_i$ is the actual value
- $\hat{y}_i$ is the predicted value

#### Root Mean Squared Error (RMSE) 👑

**Definition**: RMSE is the square root of MSE and provides a measure of how well the model predictions match the observed data. RMSE is in the same units as the target variable, making it more interpretable.

**Formula**:
$$RMSE = \sqrt{MSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

#### R-squared 🐇

Also known as Coefficient of Determination. <br>
**Definition**: R-squared measures the proportion of the variance in the dependent variable that is predictable from the independent variables. It provides an indication of the goodness of fit of a model.

**Formula**:
$$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

where:

- $\bar{y}$ is the mean of the actual values
- $\hat{y}$ is the predicted values of y
- $\sum_{i=1}^{n} (y_i - \hat{y}_i)^2$ is Residual sum of squares (RSS)
- $\sum_{i=1}^{n} (y_i - \bar{y})^2$ is Total sum of squares (TSS)

$$
R^2 = 1 - \frac{RSS}{TSS}
$$

If $R^2$ is 0.65, then we say the model explains 65% of variabillity of the dependent variable $Y$.

#### Adjusted R-squared 🐼

**Definition**: Adjusted R-squared adjusts the R-squared value for the number of predictors in the model. It accounts for the fact that adding more predictors to a model will almost always increase the R-squared value, regardless of the actual improvement in model fit.

**Formula**:
$$\text{Adjusted } R^2 = 1 - \left( \frac{(1 - R^2)(n - 1)}{n - p - 1} \right)$$

where:

- $n$ is the number of observations
- $p$ is the number of predictors

#### Example

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

### Evaluation Metrics - Classification

These metrics are commonly used in evaluating the performance of classification models. They are derived from a confusion matrix, which summarizes the predictions of a binary classifier.

#### Confusion Matrix

For a binary classification problem:

|                     | Predicted Positive  | Predicted Negative  |
| ------------------- | ------------------- | ------------------- |
| **Actual Positive** | True Positive (TP)  | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN)  |

- **True Positive (TP):** Correctly predicted positive cases.
- **True Negative (TN):** Correctly predicted negative cases.
- **False Positive (FP):** Negative cases incorrectly predicted as positive.
- **False Negative (FN):** Positive cases incorrectly predicted as negative.

---

#### Accuracy

Accuracy measures the proportion of correctly classified instances out of the total instances.

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

#### Precision (Positive Predictive Value)

Precision measures the proportion of true positive predictions out of all positive predictions.

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

#### Recall (Sensitivity)

Recall measures the proportion of actual positives correctly identified. It is also known as **True positive rate**

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

#### Specificity

Specificity measures the proportion of actual negatives correctly identified. It is also known as **True Negative Rate**

$$
\text{Specificity} = \frac{TN}{TN + FP}
$$

#### F1 Score

The F1 Score is the harmonic mean of precision and recall, balancing the two.

$$
\text{F1 Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

#### Area Under the ROC Curve (AUC-ROC)

The AUC-ROC measures the ability of a classifier to distinguish between classes across all thresholds. The Receiver Operating Characteristic (ROC) curve plots **True Positive Rate (TPR)** vs. **False Positive Rate (FPR)** for various thresholds.

- **TPR:**
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

#### Matthews Correlation Coefficient (MCC) 🔗

The MCC is a balanced metric that considers all elements of the confusion matrix, even in imbalanced datasets. It ranges from -1 to +1:

- +1: Perfect prediction.
- 0: Random guessing.
- -1: Total disagreement.

$$
\text{MCC} = \frac{(TP \cdot TN) - (FP \cdot FN)}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}
$$

#### Logarithmic Loss (Log Loss) 📉

Log Loss measures the uncertainty of predictions by penalizing incorrect confidence levels. Lower Log Loss indicates better predictions.

$$
\text{Log Loss} = -\frac{1}{N} \sum_{i=1}^{N} \left( y_i \cdot \log(p_i) + (1 - y_i) \cdot \log(1 - p_i) \right)
$$

- $y_i$: Actual label (0 or 1).
- $p_i$: Predicted probability of the positive class.
- $N$: Total number of samples.

#### Cohen's Kappa ⚖️

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

#### Example

**Dataset**

**Actual Values:** `[1, 0, 1, 1, 0, 0, 1, 0, 0, 1]`  
**Predicted Probabilities:** `[0.9, 0.3, 0.8, 0.4, 0.2, 0.1, 0.7, 0.6, 0.2, 0.9]`  
**Predicted Labels (Threshold = 0.5):** `[1, 0, 1, 0, 0, 0, 1, 1, 0, 1]`

**Confusion Matrix**

|                 | Predicted Positive | Predicted Negative |
| --------------- | ------------------ | ------------------ |
| Actual Positive | 459 (TP)           | 51 (FN)            |
| Actual Negative | 90 (FP)            | 400 (TN)           |

**Metrics Calculation**

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

**Summary Table**

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

### Regularization

**Regularization** is a technique used to prevent overfitting by adding a penalty term to the cost function. It discourages the model from fitting too closely to the training data, which improves generalization to unseen data.

1. <u>_Overfitting_</u>: Complex models with many parameters (e.g., high-dimensional regression) can overfit the training data.
2. <u>_Multicollinearity_</u>: In regression, highly correlated predictors can make coefficient estimates unstable.
3. <u>_Improved Generalization_</u>: Regularization helps simplify the model by shrinking parameters.

#### Regularized Cost Function

For a regression or classification model, the general cost function becomes:

$$
J(\boldsymbol{\beta}) = \text{Loss Function}(\boldsymbol{\beta}) + \lambda \cdot \text{Regularization Term},
$$

where:

- **Loss Function**: Measures the prediction error (e.g., Mean Squared Error for regression, Cross-Entropy for classification).
- **Regularization Term**: Penalizes large or complex parameter values (e.g., $\|\boldsymbol{\beta}\|_1$ or $\|\boldsymbol{\beta}\|_2$).
- $\lambda$: Regularization strength (hyperparameter).

#### Types of Regularization

1. **L1 Regularization (Lasso)**:

   $$
   J(\boldsymbol{\beta}) = \text{Loss Function}(\boldsymbol{\beta}) + \lambda \|\boldsymbol{\beta}\|_1,
   $$

   where $\|\boldsymbol{\beta}\|_1 = \sum_{j=1}^p |\beta_j|$.

   - Encourages sparsity by shrinking some coefficients to zero.
   - Used for feature selection.

2. **L2 Regularization (Ridge)**:

   $$
   J(\boldsymbol{\beta}) = \text{Loss Function}(\boldsymbol{\beta}) + \lambda \|\boldsymbol{\beta}\|_2^2,
   $$

   where $\|\boldsymbol{\beta}\|_2^2 = \sum_{j=1}^p \beta_j^2$.

   - Penalizes large coefficients but doesn’t force them to zero.
   - Stabilizes models in the presence of multicollinearity.

3. **Elastic Net**:

   $$
   J(\boldsymbol{\beta}) = \text{Loss Function}(\boldsymbol{\beta}) + \lambda_1 \|\boldsymbol{\beta}\|_1 + \lambda_2 \|\boldsymbol{\beta}\|_2^2.
   $$

   - Combines L1 and L2 regularization for better flexibility.

### Bias-Variance trade-off

### Cross-Validation Methods

<!-- TODO: Add notes on cross validation methods-->

## Regression Models

### Linear Regression

#### Simple Linear Regression

Linear regression with one variable, also known as Simple Linear Regression, aims to model the relationship between a single independent variable (X) and a dependent variable (Y) by fitting a linear equation to observed data. The equation of a line is:

$$
Y = \beta_0 + \beta_1X + \epsilon
$$

Where:

- $Y$ is the dependent variable.
- $X$ is the independent variable.
- $\beta_0$ is the y-intercept (constant term).
- $\beta_1$ is the slope of the line (regression coefficient).
- $\epsilon$ is the error term (residuals).

**Steps:**

1. **Data Collection**: Gather data points of $X$ and $Y$.
2. **Estimation of Coefficients**: Use methods like Ordinary Least Squares (OLS) to estimate $\beta_0$ and $\beta_1$.
3. **Model Fitting**: Fit the linear equation to the data.
4. **Prediction**: Use the linear model to predict $Y$ for given $X$ values.
5. **Evaluation**: Assess the model's accuracy using metrics like R-squared and Mean Squared Error (MSE).

**Finding $\beta_1$ using Covariance and Standard Deviation**

In Simple Linear Regression, $\beta_1$ can be found using the covariance of $X$ and $Y$, and the variance of $X$. The formula is:

$$
\beta_1 = \frac{\text{Cov}(X, Y)}{\text{Var}(X)}
$$

Alternatively, using standard deviations:

$$
\beta_1 = r \cdot \frac{s_y}{s_x}
$$

Where:

- $r$ is the Pearson correlation coefficient between $X$ and $Y$.
- $s_y$ is the standard deviation of $Y$.
- $s_x$ is the standard deviation of $X$.

---

#### Multiple Linear Regression (MLR)

Multiple Linear Regression models the relationship between two or more independent variables and a dependent variable by fitting a linear equation to observed data. The equation is:

$$
Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + ... + \beta_nX_n + \epsilon
$$

Where:

- $Y$ is the dependent variable.
- $X_1, X_2, ..., X_n$ are the independent variables.
- $\beta_0$ is the y-intercept (constant term).
- $\beta_1, \beta_2, ..., \beta_n$ are the coefficients of the independent variables.
- $\epsilon$ is the error term (residuals).

**Steps:**

1. **Data Collection**: Gather data points of $X_1, X_2, ..., X_n$ and $Y$.
2. **Estimation of Coefficients**: Use methods like OLS to estimate $\beta_0, \beta_1, \beta_2, ..., \beta_n$.
3. **Model Fitting**: Fit the linear equation to the data.
4. **Prediction**: Use the linear model to predict $Y$ for given values of $X_1, X_2, ..., X_n$.
5. **Evaluation**: Assess the model's accuracy using metrics like Adjusted R-squared, MSE, and the F-test for overall significance.

**Key Points:**

- **Assumptions**: Linearity, Independence, Homoscedasticity, Normality of errors.
- **Multicollinearity**: Check for multicollinearity using Variance Inflation Factor (VIF).
- **Feature Selection**: Use techniques like forward selection, backward elimination, and regularization (Lasso, Ridge) to select relevant features.

**$\beta$ Matrix using MLE**

In Multiple Linear Regression, to estimate the $\beta$ vector (coefficients), we use the formula:

$$
\beta = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{Y}
$$

Where:

- $\mathbf{X}^T$ is the transpose of the $\mathbf{X}$ matrix.
- $(\mathbf{X}^T \mathbf{X})^{-1}$ is the inverse of the product of $\mathbf{X}^T$ and $\mathbf{X}$.
- $\mathbf{X}^T \mathbf{Y}$ is the product of the transpose of $\mathbf{X}$ and $\mathbf{Y}$.

**$\beta$ Matrix using correaltion coefficients**

To calculate the betas (regression coefficients) in Multiple Linear Regression (MLR) using correlations and standard deviations, you can follow these steps:

**Step-by-Step Calculation**

1. **Standardize the Variables**:
   Standardize each independent variable $X_i$ and the dependent variable $Y$ to have a mean of 0 and a standard deviation of 1. This removes the units and allows for easier calculation of correlations.

   Standardized variable $Z$ for $X_i$ and $Y$ is calculated as:

   $$
   Z_{X_i} = \frac{X_i - \bar{X_i}}{s_{X_i}}
   $$

   $$
   Z_Y = \frac{Y - \bar{Y}}{s_Y}
   $$

2. **Calculate the Correlation Coefficients**:
   Compute the Pearson correlation coefficients $r_{X_i,Y}$ for each independent variable $X_i$ with the dependent variable $Y$.

3. **Form the Correlation Matrix (R)**:
   Create a matrix $R$ of the correlation coefficients between the independent variables. For example, for three variables $X_1, X_2, X_3$:

   $$
   R = \begin{pmatrix}
   1 & r_{X_1X_2} & r_{X_1X_3} \\
   r_{X_2X_1} & 1 & r_{X_2X_3} \\
   r_{X_3X_1} & r_{X_3X_2} & 1
   \end{pmatrix}
   $$

4. **Form the Correlation Vector (r)**:
   Create a vector $r$ of the correlation coefficients between each independent variable $X_i$ and the dependent variable $Y$:

   $$
   r = \begin{pmatrix}
   r_{X_1Y} \\
   r_{X_2Y} \\
   r_{X_3Y}
   \end{pmatrix}
   $$

5. **Calculate the Beta Coefficients**:
   Compute the beta coefficients using the formula:
   $$ \beta = R^{-1} r $$
   Where $R^{-1}$ is the inverse of the correlation matrix $R$.

**Example**

Suppose we have three independent variables $X_1, X_2, X_3$ and a dependent variable $Y$. The correlation coefficients are as follows:

- $r_{X_1Y} = 0.6$
- $r_{X_2Y} = 0.5$
- $r_{X_3Y} = 0.7$
- $r_{X_1X_2} = 0.4$
- $r_{X_1X_3} = 0.3$
- $r_{X_2X_3} = 0.5$

The correlation matrix $R$ and vector $r$ are:

$$
R = \begin{pmatrix}
1 & 0.4 & 0.3 \\
0.4 & 1 & 0.5 \\
0.3 & 0.5 & 1
\end{pmatrix}
$$

$$
r = \begin{pmatrix}
0.6 \\
0.5 \\
0.7
\end{pmatrix}
$$

Then calculate the beta coefficients:

$$
\beta = R^{-1} r
$$

**Note:** Ensure to perform matrix inversion and multiplication carefully.

#### HAT Matrix

$$
\hat{Y} = X\beta \\
\hat{Y} = X(X^TX)^{-1}X^TY
$$

The hat matrix, also known as the projection matrix or influence matrix, plays a key role in linear regression. It maps the observed values to the predicted values. Here are the important properties of the hat matrix.

**Properties of the Hat Matrix**

1. **Definition**:
   The hat matrix $\mathbf{H}$ is defined as:

$$
 \mathbf{H} = \mathbf{X} (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T
$$

Where:

- $\mathbf{X}$ is the matrix of independent variables (including a column of ones for the intercept term).
- $\mathbf{X}^T$ is the transpose of $\mathbf{X}$.
- $(\mathbf{X}^T \mathbf{X})^{-1}$ is the inverse of $\mathbf{X}^T \mathbf{X}$.

2. **Idempotent**:
   The hat matrix is idempotent, meaning:

$$
 \mathbf{H}^2 = \mathbf{H}
$$

3. **Symmetric**:
   The hat matrix is symmetric, meaning:

$$
 \mathbf{H} = \mathbf{H}^T
$$

4. **Eigenvalues**:
   The eigenvalues of the hat matrix are either 0 or 1.

5. **Trace**:
   The trace of the hat matrix (the sum of its diagonal elements) is equal to the rank of the matrix $\mathbf{X}$, which is the number of independent variables (including the intercept term). If there are $p$ predictors, then:

$$
 \text{trace}(\mathbf{H}) = p
$$

6. **Projection**:
   The hat matrix projects the observed $\mathbf{Y}$ onto the column space of $\mathbf{X}$. It transforms the observed values into the predicted values:

$$
 \hat{\mathbf{Y}} = \mathbf{H} \mathbf{Y}
$$

7. **Influence**:
   The diagonal elements of the hat matrix, $h_{ii}$, are measures of the leverage of the $i$-th observation. Higher leverage points have more influence on the regression fit.

- The term "hat matrix" comes from its role in putting a "hat" on $\mathbf{Y}$ to indicate predicted values $\hat{\mathbf{Y}}$.
- In diagnostics, high-leverage points (high $h_{ii}$ values) can be scrutinized for their potential influence on the model. Points with high leverage but small residuals may still significantly affect the regression coefficients.

Understanding these properties can help you analyze and interpret the results of a regression model more effectively.

### Ridge Regression

Ridge Regression is a type of linear regression that includes a regularization term to prevent overfitting. It adds a penalty term to the cost function, which is proportional to the square of the magnitude of the coefficients. The objective is to minimize the following cost function:

The general regression model is:

$$
\mathbf{y} = \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\epsilon}.
$$

The regularized cost function is:

$$
J(\boldsymbol{\beta}) = \|\mathbf{y} - \mathbf{X} \boldsymbol{\beta}\|^2 + \lambda R(\boldsymbol{\beta}),
$$

$$
J(\boldsymbol{\beta}) = \|\mathbf{y} - \mathbf{X} \boldsymbol{\beta}\|^2 + \lambda \|\boldsymbol{\beta}\|_2^2,
$$

where $\|\boldsymbol{\beta}\|_2^2 = \sum_{j=1}^p \beta_j^2$.

- Penalizes large coefficients but doesn’t force them to zero.
- Stabilizes models in the presence of multicollinearity.
  where $R(\boldsymbol{\beta})$ is the regularization term.

$$
\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^\top \mathbf{y}.
$$

This is the closed form equation to find $\beta$ matrix. In other cases where it's hard/impossible to find a closed form to $\beta$ matrix we use iterative methods like gradient descent to find the global minima of for the cost function.

**Gradient Descent**

$$

J(\beta) = \frac{1}{2m} \sum_{i=1}^{m} (y_i - \mathbf{x}_i^T \beta)^2 + \lambda \sum_{j=1}^{p} \beta_j^2


$$

Where:

- $m$ is the number of training examples.
- $y_i$ is the actual value of the $i$-th training example.
- $\mathbf{x}_i$ is the feature vector of the $i$-th training example.
- $\beta$ is the vector of regression coefficients.
- $\lambda$ is the regularization parameter.

Gradient descent is an optimization algorithm used to minimize the cost function by iteratively updating the coefficients. The steps involved are:

- Initialize the coefficient vector $\beta$ with zeros or small random values.

- Calculate the gradient of the cost function with respect to the coefficients. For Ridge Regression, the gradient is:

$$

\nabla J(\beta) = -\frac{1}{m} \mathbf{X}^T (\mathbf{Y} - \mathbf{X} \beta) + 2\lambda \beta


$$

Where:

- $\mathbf{X}$ is the matrix of input features.
- $\mathbf{Y}$ is the vector of target values.
- $\mathbf{X}^T$ is the transpose of $\mathbf{X}$.

Update the Coefficients<br>
Update the coefficients using the gradient descent update rule:

$$
 \beta := \beta - \alpha \nabla J(\beta)
$$

Where:

- $\alpha$ is the learning rate.

1. **Initialize** the coefficient vector $\beta$ (e.g., with zeros).
2. **Repeat** until convergence:

$$
 \hat{\mathbf{Y}} = \mathbf{X} \beta
$$

$$
 \mathbf{E} = \mathbf{Y} - \hat{\mathbf{Y}}
$$

$$
 \nabla J(\beta) = -\frac{1}{m} \mathbf{X}^T \mathbf{E} + 2\lambda \beta
$$

$$
 \beta := \beta - \alpha \left( -\frac{1}{m} \mathbf{X}^T \mathbf{E} + 2\lambda \beta \right)
$$

$$
 \beta := \beta + \alpha \left( \frac{1}{m} \mathbf{X}^T \mathbf{E} - 2\lambda \beta \right)
$$

### Logistic Regression

Logistic Regression is used for binary classification problems, where the outcome $Y$ is a binary variable (0 or 1). Instead of modeling $Y$ directly, Logistic Regression models the probability that $Y = 1$ as a function of the input variables $X$. The model uses the logistic (sigmoid) function to ensure that the output is between 0 and 1.

The logistic function is given by:

$$
 \sigma(z) = \frac{1}{1 + e^{-z}}
$$

For Logistic Regression, $z$ is a linear combination of the input features:

$$
 z = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_n X_n
$$

The probability that $Y = 1$ given $X$ is:

$$
 P(Y = 1 | X) = \sigma(\beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_n X_n)
$$

**Loss Function (Log Loss)**

The loss function for Logistic Regression is the log loss (also known as logistic loss or binary cross-entropy). For a single training example, the log loss is defined as:

$$
 \text{Log Loss}(y, \hat{y}) = -[y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})]
$$

Where:

- $y$ is the actual label (0 or 1).
- $\hat{y}$ is the predicted probability that $y = 1$.

**Cost Function**

The cost function for Logistic Regression is the average log loss over all training examples. For $m$ training examples, the cost function $J(\beta)$ is:

$$
 J(\beta) = -\frac{1}{m} \sum_{i=1}^m [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]
$$

Where:

- $y_i$ is the actual label for the $i$-th training example.
- $\hat{y}_i$ is the predicted probability for the $i$-th training example.

To minimize the cost function, we use gradient descent. The gradient of the cost function with respect to the coefficients $\beta$ is:

$$
 \nabla J(\beta) = \frac{1}{m} \mathbf{X}^T (\sigma(\mathbf{X} \beta) - \mathbf{Y})
$$

Where:

- $\mathbf{X}$ is the matrix of input features.
- $\beta$ is the vector of coefficients.
- $\mathbf{Y}$ is the vector of actual labels.
- $\sigma(\mathbf{X} \beta)$ is the vector of predicted probabilities.

1. **Initialize Coefficients**: Initialize the coefficient vector $\beta$ with zeros or small random values.
2. **Repeat Until Convergence**:

$$
 \hat{\mathbf{Y}} = \sigma(\mathbf{X} \beta)
$$

$$
 \mathbf{E} = \hat{\mathbf{Y}} - \mathbf{Y}
$$

$$
 \nabla J(\beta) = \frac{1}{m} \mathbf{X}^T \mathbf{E}
$$

$$
 \beta := \beta - \alpha \nabla J(\beta)
$$

**Example**

Let's consider an example with a single feature $X$ and binary target $Y$.

$$
 \beta = \begin{pmatrix}
 \beta_0 \\
 \beta_1
 \end{pmatrix}
$$

$$
 \hat{Y} = \sigma(\beta_0 + \beta_1 X)
$$

$$
 E = \hat{Y} - Y
$$

$$
 \nabla J(\beta) = \frac{1}{m} \sum_{i=1}^m X_i^T ( \hat{Y}_i - Y_i)
$$

$$
 \beta := \beta - \alpha \nabla J(\beta)
$$

By iteratively applying these steps, you can optimize the coefficients to minimize the cost function and improve the model's performance.

## K-Nearest Neighbors (KNN) Algorithm

The K-Nearest Neighbors (KNN) algorithm is a foundational technique in supervised learning, widely known for its simplicity and effectiveness. It can be used for both classification and regression tasks. Here, we'll focus on explaining the KNN algorithm for classification in great detail.

1. **Supervised Learning**:

   - **Training Data**: A set of labeled examples where each example consists of an input feature vector and a corresponding label.
   - **Test Data**: A set of examples used to evaluate the performance of the model.

2. **Instance-Based Learning**:

   - KNN is an instance-based learning algorithm (also known as a lazy learner) because it does not build an explicit model during the training phase. Instead, it makes predictions using the entire training dataset.

3. **Distance Metric**:

   - The algorithm relies on a distance metric to measure the similarity between instances. Common distance metrics include:

     1. **Euclidean Distance**

     The Euclidean distance between two points $\mathbf{x} = (x_1, x_2, ..., x_n)$ and $\mathbf{y} = (y_1, y_2, ..., y_n)$ in an n-dimensional space is given by:

     $$
      d(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
     $$

     2. **Manhattan Distanc**e

     The Manhattan distance (also known as L1 distance or city block distance) between two points $\mathbf{x} = (x_1, x_2, ..., x_n)$ and $\mathbf{y} = (y_1, y_2, ..., y_n)$ is given by:

     $$
      d(\mathbf{x}, \mathbf{y}) = \sum_{i=1}^{n} |x_i - y_i|
     $$

     3. **Minkowski Distance**

     The Minkowski distance is a generalized form of both Euclidean and Manhattan distances. For two points $\mathbf{x} = (x_1, x_2, ..., x_n)$ and $\mathbf{y} = (y_1, y_2, ..., y_n)$, and a parameter $p$, the Minkowski distance is given by:

     $$
      d(\mathbf{x}, \mathbf{y}) = \left( \sum_{i=1}^{n} |x_i - y_i|^p \right)^{\frac{1}{p}}
     $$

     - When $p = 1$, Minkowski distance is equivalent to Manhattan distance.
     - When $p = 2$, Minkowski distance is equivalent to Euclidean distance.

4. **Parameter $k$**:
   - The parameter $k$ represents the number of nearest neighbors to consider when making a prediction. The choice of $k$ can significantly impact the algorithm's performance.

**STEPS**

- Each data point is represented by a feature vector, and the entire dataset is stored in memory.
- When a new, unseen data point needs to be classified, the algorithm calculates the distance between this new point and all the points in the training dataset using the chosen distance metric.

- The algorithm identifies the $k$ points in the training dataset that are closest to the new data point. These $k$ points are referred to as the "nearest neighbors."

- For classification, KNN employs a voting mechanism. The class labels of the $k$ nearest neighbors are considered, and the new data point is assigned to the class that appears most frequently among the neighbors.

- The final classification of the new data point is determined based on the majority vote of its nearest neighbors.

**Considerations and Challenges**

1. **Choice of $k$**:

   - Selecting the appropriate value of $k$ is crucial. A small value of $k$ can lead to high variance (overfitting), while a large value of $k$ can lead to high bias (underfitting). Cross-validation can help determine the optimal $k$.

2. **Distance Metric**:

   - The choice of distance metric can affect the performance of the algorithm. The Euclidean distance is commonly used, but other metrics may be more appropriate depending on the data.

3. **Computational Complexity**:

   - Since KNN stores all the training data and performs distance calculations for each prediction, it can be computationally expensive for large datasets.

4. **Feature Scaling**:

   - Features with different scales can impact the distance calculation. Therefore, it's essential to normalize or standardize the features before applying the KNN algorithm.

5. **Handling Missing Data**:

   - Missing values in the dataset can pose challenges. Imputation methods or excluding instances with missing values are common strategies.

6. **Curse of Dimensionality**:
   - In high-dimensional spaces, the concept of distance becomes less meaningful, and the algorithm's performance may degrade. Dimensionality reduction techniques can be used to mitigate this issue.

KNN is a versatile and straightforward algorithm that makes predictions based on the similarity between data points. It is particularly effective for small to medium-sized datasets with a relatively low number of features. However, careful consideration of the parameter $k$, distance metric, and feature scaling is essential to ensure optimal performance.

## Descision Trees

A decision tree is a popular algorithm used in machine learning for classification and regression tasks. It's a tree-like model of decisions, where each internal node represents a test on an attribute, each branch represents an outcome of the test, and each leaf node represents a class label (in classification) or a continuous value (in regression).

**Decision Tree Structure**

1. **Root Node**: The top node of the tree. It represents the entire dataset, which is then split into subsets.
2. **Internal Nodes**: These nodes represent the attributes on which the data is split.
3. **Branches**: These represent the outcomes of a decision or test.
4. **Leaf Nodes**: The terminal nodes that predict the outcome.

**Key Concepts and Formulas**

1. **Gini Index**: A measure of impurity or diversity used in classification trees.

   - Formula:
     $$
     Gini = 1 - \sum_{i=1}^{c} p_i^2
     $$
     where $p_i$ is the probability of an element being classified into a particular class.

2. **Entropy**: Used in building decision trees using the ID3 algorithm. It measures the amount of uncertainty or impurity.

   - Formula:

     $$
     Entropy(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)
     $$

     where $p_i$ is the probability of an element belonging to class $i$, and $c$ is the number of classes.

   - Entropy = 0 in homogenous case
   - Max it can be $\text{Entropy} = log_2 (\text{No. of Classes})$

3. **Information Gain**: Measures the reduction in entropy or impurity. It is the difference between the entropy before and after a split.

   - Formula:
     $$
     IG(T, X) = Entropy(T) - \sum_{i=1}^{n} \frac{|T_i|}{|T|} Entropy(T_i)
     $$
     where $T$ is the set of data, $X$ is the attribute, $n$ is the number of partitions (values) of the attribute $X$, and $|T_i|$ is the number of elements in partition $i$.

4. **Chi-Square**: A statistical method used to measure the significance of the differences between sub-nodes and parent nodes.

   - Formula:
     $$
     \chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}
     $$
     where $O_i$ is the observed frequency, and $E_i$ is the expected frequency.

5. **Reduction in Variance**: Used for regression trees. It measures the variance of the target variable in the nodes.
   - Formula:
     $$
     Var(T) = \frac{1}{|T|} \sum_{i=1}^{|T|} (y_i - \bar{y})^2
     $$
     where $y_i$ is the actual value, $\bar{y}$ is the mean of the values, and $|T|$ is the number of observations in the node.

**Building a Decision Tree**

1. **Select Attribute for Root Node**: Calculate Gini, Entropy, Information Gain, or Chi-Square for all attributes and select the attribute with the best score.
2. **Split the Dataset**: Partition the dataset based on the selected attribute.
3. **Repeat Process**: For each partition, repeat the process to create internal nodes.
4. **Stop Condition**: Stop when all instances in a node belong to the same class, or when splitting no longer adds value to the predictions.

### Example

| Outlook  | Temperature | Humidity | Windy | PlayTennis |
| -------- | ----------- | -------- | ----- | ---------- |
| Sunny    | Hot         | High     | False | No         |
| Sunny    | Hot         | High     | True  | No         |
| Overcast | Hot         | High     | False | Yes        |
| Rain     | Mild        | High     | False | Yes        |
| Rain     | Cool        | Normal   | False | Yes        |
| Rain     | Cool        | Normal   | True  | No         |
| Overcast | Cool        | Normal   | True  | Yes        |
| Sunny    | Mild        | High     | False | No         |
| Sunny    | Cool        | Normal   | False | Yes        |
| Rain     | Mild        | Normal   | False | Yes        |
| Sunny    | Mild        | Normal   | True  | Yes        |
| Overcast | Mild        | High     | True  | Yes        |
| Overcast | Hot         | Normal   | False | Yes        |
| Rain     | Mild        | High     | True  | No         |

1. Calculate Initial Entropy

We start by calculating the entropy of the entire dataset. Entropy measures the impurity or disorder in the dataset.

$$
Entropy(S) = - \sum\_{i=1}^{c} p_i \log_2(p_i)
$$

Here, $p_i$ is the probability of each class (Yes or No) in the dataset.

- Probability of "Yes" ($p_1$) = 9/14
- Probability of "No" ($p_2$) = 5/14

$$
Entropy(S) = - \left( \frac{9}{14} \log_2 \left( \frac{9}{14} \right) + \frac{5}{14} \log_2 \left( \frac{5}{14} \right) \right) \approx 0.94
$$

2. Calculate Entropy for Each Attribute and Split

Next, we calculate the entropy for each attribute (Outlook, Temperature, Humidity, Windy) to determine the best attribute for splitting the data.

**Example: Outlook**

- **Sunny**: 5 instances (2 Yes, 3 No)
- **Overcast**: 4 instances (4 Yes, 0 No)
- **Rain**: 5 instances (3 Yes, 2 No)

$$
Entropy(Sunny) = - \left( \frac{2}{5} \log_2 \left( \frac{2}{5} \right) + \frac{3}{5} \log_2 \left( \frac{3}{5} \right) \right) \approx 0.97
$$

$$
Entropy(Overcast) = - \left( \frac{4}{4} \log_2 \left( \frac{4}{4} \right) + \frac{0}{4} \log_2 \left( \frac{0}{4} \right) \right) = 0
$$

$$
Entropy(Rain) = - \left( \frac{3}{5} \log_2 \left( \frac{3}{5} \right) + \frac{2}{5} \log_2 \left( \frac{2}{5} \right) \right) \approx 0.97
$$

3. Calculate Information Gain

Information Gain measures the reduction in entropy after a dataset is split on an attribute.

$$
IG(T, X) = Entropy(T) - \sum\_{i=1}^{n} \frac{|T_i|}{|T|} Entropy(T_i)
$$

**For Outlook:**

$$
IG(S, Outlook) = 0.94 - \left( \frac{5}{14} \cdot 0.97 + \frac{4}{14} \cdot 0 + \frac{5}{14} \cdot 0.97 \right) \approx 0.25
$$

4. Select the Attribute with the Highest Information Gain

We repeat the calculations for all attributes and select the one with the highest Information Gain. In this case, let's assume "Outlook" has the highest Information Gain.

5. Split the Dataset

We split the dataset based on the chosen attribute (Outlook) and repeat the process for each subset.

- **Sunny** subset
- **Overcast** subset
- **Rain** subset

6. Repeat the Process for Each Subset

For each subset, we repeat steps 1 to 5 until we reach a leaf node (where all instances belong to the same class) or splitting no longer improves the prediction.

### Example Decision Tree

Here's a simplified decision tree based on the calculations:

```
                     Outlook
                        |
            ־־־־־־־־־־־־־־־־־־־־־־־־
            |          |           |
          Sunny    Overcast       Rain
            |          |           |
           NO         YES         YES
            |         (leaf)       |
        HUMIDITY              ־־־־־־־־־־־
            |                 |         |
      ־־־־־־־־־־־־־          No        Yes
      |           |         (leaf)    (leaf)
     High       Normal
      |           |
     No          Yes
    (Leaf)      (Leaf)
```

By using these steps and calculations, we build a decision tree that helps us make predictions based on the attributes of the data. If you have any specific questions or need more details, feel free to ask!

   <!--### Estimating Parameters-->
   <!---->
   <!--- **Lasso Regression (L1 Regularization)**:-->
   <!--  The optimization involves minimizing an objective with an L1 penalty. This is solved using techniques like:-->
   <!---->
   <!--  - Coordinate Descent,-->
   <!--  - Least Angle Regression (LARS).-->
   <!---->
   <!--- **Elastic Net**:-->
   <!--  Solved using numerical methods such as gradient descent or coordinate descent.-->
   <!---->
   <!--### Classification Models-->
   <!---->
   <!--For classification, regularization is added to the cost function, often based on **Cross-Entropy Loss**:-->
   <!---->
   <!--$$-->
   <!--J(\boldsymbol{\beta}) = -\frac{1}{n} \sum_{i=1}^n \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right] + \lambda R(\boldsymbol{\beta}).-->
   <!--$$-->
   <!---->
   <!--- **Logistic Regression with L2 Regularization**:-->
   <!--  Solved using:-->
   <!---->
   <!--  - Gradient Descent,-->
   <!--  - Newton’s Method.-->
   <!---->
   <!--2. **Elastic Net Parameters**:-->
   <!--   - $\alpha$: Determines the mix between L1 and L2 regularization.-->
