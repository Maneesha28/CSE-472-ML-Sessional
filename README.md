# CSE-472-Machine-Learning-Sessional

This repository contains all the assignments of CSE 472: Machine Learning Sessional course. An overview of the assignments is given below. For more details, please refer to the respective assignment folders.

## Assignment-1 (Linear Algebra)

In this assignment, we implemented the following concepts of Linear Algebra using `numpy` library in Python:

1. Matrix Transformation

2. Eigen Decomposition

    2.1 Random Matrix

    2.2 Symmetric Matrix
    
3. Image Reconstruction using Singular Value Decomposition (SVD)


## Assignment-2 (Linear Regression)

In this assignment, we implemented a Logistic Regression (LR) classifier and used it within AdaBoost algorithm.

We also implemented different data preprocessing techniques for the given datasets. Finally, we compared the performance (`Accuracy`, `TPR`, `TNR`, `Precision`, `False Discovery Rate`, `F1 Score`) of the LR classifier with and without AdaBoost. 

## Assignment-3 (Feed Forward Neural Network)

In this assignment, we built an FNN from scratch and applied our FNN to classify letters (`EMNIST` dataset containing 28x28 images of letters from the Latin alphabet).

Basic components of the FNN:
- Dense Layer
- ReLU Activation Layer
- Dropout Layer
- Softmax Activation Layer

We implemented the [backpropagation algorithm](https://towardsdatascience.com/deriving-backpropagation-with-cross-entropy-loss-d24811edeaf9) to train the model. The weights were updated using mini-batch gradient descent.

Libraries used: `opencv`, `pillow`, `torchvision`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `tqdm`, `pickle`, `scipy`, `sklearn`.

<ins>*No deep learning framework was allowed for this assignment*</ins>.

## Assignment-4 (PCA and EM Algorithm)

In this assignment, we implemented the following concepts: 
1. [Principal Component Analysis (PCA)](https://cs229.stanford.edu/notes2020spring/cs229-notes10.pdf): a useful tool used in machine learning and data science to simplify the complexity of high dimensional data while retaining its trends and patterns. It does so by transforming the data into fewer dimensions, which act as summaries of features. 

2. [Expectation-Maximization (EM) algorithm](https://cs229.stanford.edu/notes2020spring/cs229-notes8.pdf): to estimate the Gaussian Mixture Model (GMM), which is a widely used unsupervised clustering method to group a set of sample data into clusters.

