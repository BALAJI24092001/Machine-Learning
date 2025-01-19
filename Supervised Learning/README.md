# SUPERVISED LEARNING

Supervised Machine Learning is where models are trained on data to predict something. The data model is built in such a way that the predicted values and original values are as close/similar as possible.
There are two types of prediction models, where we predict either continuous outcome(Regression) or classifying an observation into a suitable category.(Classification).

Supervised learning is a machine learning approach where a model is trained on labeled data. Here's a list of commonly used models in supervised learning, categorized into different types:

---

### CONTENTS

###### Linear Models

1. Linear Regression (for regression tasks)
2. Logistic Regression (for binary classification)
3. Ridge Regression (L2 regularization)
4. Lasso Regression (L1 regularization)
5. Elastic Net Regression (combination of L1 and L2 regularization)
6. Generalized Linear Models(GLM)

###### Support Vector Machines (SVMs)

1. Linear SVM (for linearly separable data)
2. Kernel SVM (for non-linear data, using kernels like RBF, polynomial)

###### Decision Tree-Based Models

1. Decision Trees (for classification and regression)
2. Random Forest (ensemble of decision trees)
3. Gradient Boosting Machines (GBM):
   - XGBoost
   - LightGBM
   - CatBoost
4. AdaBoost
5. Extra Trees (Extremely Randomized Trees)

###### Bayesian Models

1. Naive Bayes:
   - Gaussian Naive Bayes
   - Multinomial Naive Bayes
   - Bernoulli Naive Bayes
2. Bayesian Linear Regression

###### Instance-Based Learning

1. k-Nearest Neighbors (k-NN)

###### Neural Networks (In [Deep Learning](../DeepLearning/README.md))

1. Multilayer Perceptrons (MLPs) (basic feedforward neural networks)
2. Convolutional Neural Networks (CNNs) (if the task involves image data)
3. Recurrent Neural Networks (RNNs) (if the task involves sequence data)
4. Transformer-based Models (for tasks like text classification, e.g., BERT)

###### Misc Topics

1. [Cost Minimization and loss functions](#computing-cost-for-regression-models)
2. Gradient Descent
3. Stochastic Gradient Descent
4. Bias Variance trade-off
5. Regularization
6. Data transformations
7. Cross-Validation methods

---

### Computing Cost for Regression Models

$loss = e_{i} = \hat{y_{i}} - y_i$.

where $y_i$ is observed value of the dependent variable and $\hat{y_i}$ is the predicted value.

Cost is a measure of how well our model is predicting.
Loss is a how much error we get for a single observation, whereas the cost is the average of all the loss.

The equation for cost with one variable is:

$$
\begin{align}
J(w,b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})^2
\end{align}
$$

where

$$
\begin{align}
f_{w,b}(x^{(i)}) = wx^{(i)} + b
\end{align}
$$

- $f_{w,b}(x^{(i)})$ is our prediction for example $i$ using parameters $w,b$.
- $(f_{w,b}(x^{(i)}) - y^{(i)})^2$ is the squared difference between the target value and the prediction.
- These differences are summed over all the $m$ examples and divided by `2m` to produce the cost, $J(w,b)$.  
  The cost is a measure of how accurate the model is on the training data. The cost equation (1) above shows that if $w$ and $b$ can be selected such that the predictions $f_{w,b}(x)$ match the target data $y$, the $(f_{w,b}(x^{(i)}) - y^{(i)})^2$ term will be zero and the cost minimized.
