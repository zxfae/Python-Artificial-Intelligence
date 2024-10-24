# Linear Regression Guide

## Table of Contents
- [Linear Regression Guide](#linear-regression-guide)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [What is Linear Regression?](#what-is-linear-regression)
  - [Formula](#formula)
  - [Components](#components)
    - [Dependent Variable (Y)](#dependent-variable-y)
    - [Independent Variable (X)](#independent-variable-x)
    - [Coefficient (Coef\_)](#coefficient-coef_)
      - [Examples:](#examples)
    - [Intercept](#intercept)
  - [Model Evaluation](#model-evaluation)
    - [Score (R-squared)](#score-r-squared)
      - [Interpretation:](#interpretation)
    - [Mean Squared Error (MSE)](#mean-squared-error-mse)
  - [Combining Score and MSE](#combining-score-and-mse)

## Introduction
This guide provides an overview of linear regression, its components, and methods for evaluating model performance.

## What is Linear Regression?
Linear regression is a statistical method used to model the relationship between a dependent variable and one or more independent variables by fitting a linear equation to observed data. It's particularly useful for predicting a continuous outcome variable based on one or more predictor variables. This technique is commonly employed in various fields such as economics, finance, and social sciences for forecasting, analyzing trends, and understanding the impact of different factors on a specific outcome.

## Formula
The basic formula for linear regression is:
```
y = mx + b
```
Where:
- y is the dependent variable
- x is the independent variable
- m is the coefficient (coef_)
- b is the intercept

## Components

### Dependent Variable (Y)
- This is the variable we're trying to predict or explain.

### Independent Variable (X)
- This is the variable used to predict the dependent variable.

### Coefficient (Coef_)
The coefficient represents the slope of the regression line:
```
m > 0: Y increases as X increases (positive relationship)
m < 0: Y decreases as X increases (negative relationship)
m == 0: Y doesn't change when X changes (no linear relationship)

|m| > 1: Change in Y is greater than the change in X
|m| < 1: Change in Y is smaller than the change in X
|m| = 1: Change in Y equals the change in X
```

#### Examples:
```
m = 2: Y increases by 2 units when X increases by 1 unit
m = 0.5: Y increases by 0.5 units when X increases by 1 unit
m = -1: Y decreases by 1 unit when X increases by 1 unit
```

### Intercept
The intercept represents the Y value when X equals 0.

Process to find the best-fitting line:
1. Calculate error for each point
2. Square each error (makes all errors positive)
3. Sum all squared errors
4. Find line (adjust m and b) that minimizes this sum

## Model Evaluation

### Score (R-squared)
The score, also known as the coefficient of determination or R-squared (R²), measures how well the model fits the data.

``` python
Range: 0 ≤ R² ≤ 1
R² close to 1: The model explains a large portion of the variability in the data
R² close to 0: The model explains little of the variability in the data
```

#### Interpretation:
``` python
R² = 0.75: 75% of the variability in Y is explained by X
R² = 0.50: 50% of the variability in Y is explained by X
R² = 0.90: 90% of the variability in Y is explained by X
```

### Mean Squared Error (MSE)
The MSE is used to evaluate the model's prediction accuracy:

1. Lower MSE indicates better model performance.
2. MSE = 0 means the model's predictions are perfect (rarely achieved in practice).

## Combining Score and MSE
This table represents the approximate correlation between Score (R²) and MSE:

| R² | MSE | Interpretation |
|:---:|:---:|:---|
| > 0.9 | Close to 0 | Excellent model, very accurate predictions |
| > 0.7 | Low | Good model, accurate predictions |
| 0.5 - 0.7 | Moderate | Fair model, acceptable predictions |
| < 0.5 | High | Weak model, unreliable predictions |