# Understanding Classification with Scikit-Learn

## Table of Contents
- [Understanding Classification with Scikit-Learn](#understanding-classification-with-scikit-learn)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [What is Classification?](#what-is-classification)
  - [Logistic Regression](#logistic-regression)
    - [Using :](#using-)
      - [Solver Comparison](#solver-comparison)
  - [Sigmoid](#sigmoid)
    - [Version](#version)
      - [1. Standard Sigmoid](#1-standard-sigmoid)
      - [2. Stretched Sigmoid](#2-stretched-sigmoid)
      - [3. Compressed Sigmoid](#3-compressed-sigmoid)
    - [About Coefficient (Coef\_) in Logistic Regression](#about-coefficient-coef_-in-logistic-regression)

## Introduction
Classification is a fundamental task in machine learning where we aim to categorize data into predefined classes. This section covers both binary and multi-class classification using Scikit-Learn, exploring key concepts like logistic regression, sigmoid functions, and decision boundaries

## What is Classification?
Classification is a supervised learning technique in machine learning used to categorize data into predefined classes or groups. It involves building a model that can distinguish between different classes based on input features. This method is crucial for tasks such as spam detection, medical diagnosis, and image recognition. Classification can be binary (two classes) or multi-class (more than two classes) and employs various algorithms like logistic regression, decision trees, and support vector machines to make predictions.

## Logistic Regression

### Using :
```
LogisticRegression(random_state=n, solver='algorithm', max_iter=n).fit(X,y)

X_predict = np.array([[0.5]])
prediction = model.predict(X_predict)

X_predict = np.array([[0.5]])
probabilities = model.predict_proba(X_predict)
```
#### Solver Comparison

| Solver | Strengths | Limitations | When to Use |
|:---:|:---:|:---:|:---|
| lbfgs | Fast, efficient | May fail for unstable problems | Default choice for most problems |
| newton-cg | Good for high dimensions | Slow for small datasets | High-dimensional data requiring accuracy |
| liblinear | Efficient for small data | Less suitable for large data | Smaller datasets, faster solutions needed |
| sag/saga | Efficient for large data | Less accurate for small data | Very large datasets where others are slow |
| sgd | Good for online learning | Can be unstable | Online learning, extremely large datasets |

## Sigmoid

### Version

On my version. We have three version :

#### 1. Standard Sigmoid
The standard sigmoid function, which transforms input values into a range between 0 and 1. It has a balanced transition centered at x = 0.
```
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```
#### 2. Stretched Sigmoid
A stretched version of the sigmoid function. It provides a more gradual transition over a wider range of input values. The function is shifted to the left and has a gentler slope, allowing for more nuanced output in the middle range.
``` 
def sigmoid1(x):
    return 1 / (1 + np.exp(-(0.5*x + 3)))
```
#### 3. Compressed Sigmoid
A compressed version of the sigmoid function. It features a very steep transition, creating an almost binary output for a narrow range of input values. The function is significantly shifted to the left and has a much steeper slope, resulting in rapid transitions from low to high output values.
```
def sigmoid2(x):
    return 1 / (1 + np.exp(-(5*x + 11)))
```

### About Coefficient (Coef_) in Logistic Regression

The coefficient in logistic regression represents the change in the log-odds of the outcome for a one-unit increase in the predictor variable:

Interpretation:
- The sign of the coefficient indicates the direction of the relationship.
- The magnitude of the coefficient indicates the strength of the effect.
- Exponential of the coefficient (e^coef) gives the odds ratio for a one-unit increase in X.
- Larger absolute values of the coefficient result in steeper sigmoid curves and more abrupt transitions in predicted probabilities.