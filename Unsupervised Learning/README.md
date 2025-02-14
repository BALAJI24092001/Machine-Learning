# UN-SUPERVISED LEARNING

An Un-supervised learning model is used to find patterns in data. It helps to find clusters in the data, and also has many data dimensionality reduction techniques to reduce the dimensions of the training data with minimal loss of variability.

---

### Contents

##### Clustering Algorithms

1. K-mean Clustering
2. Hierarchical Clustering
3. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
4. Gaussian Mixture Models

##### Dimentionality Reduction techniques

1. Principal Component Analysis (PCA)
2. Kernel PCA
3. t-Distributed Stochastic Neighbor Embedding (t-SNE)
4. Uniform Manifold Approximation and Projection (UMAP)
5. Factor Analysis
6. Non-Negative Matrix Factorization (NMF)
7. Independent Component Analysis (ICA)

##### Anomaly Detection

1. Isolation Forest
2. One-Class SVM
3. Autoencoders (in deep learning settings)
4. Elliptic Envelope

---

## Dimentionality Reduction

### Principal Component Analysis

> [REF](https://jonathan-hui.medium.com/machine-learning-singular-value-decomposition-svd-principal-component-analysis-pca-1d45e885e491) <br> [Sebastian Raschka](https://sebastianraschka.com/Articles/2015_pca_in_3_steps.html)

### Linear Discriminant Analysis

This is a dimension reduction technique simillar to PCA, but in PCA we try to increase the variance of only the independent variables, whereas in LDA we try to maximize the distance between the different class means and also reduce the variance in the independent variables for each class there by making the classes get denser and the means of the classes dispersed as much as possible.

Let us assume we have n observation of k classes. Then we need to calculate

> n [Sebastian Raschaka](https://sebastianraschka.com/Articles/2014_python_lda.html)
