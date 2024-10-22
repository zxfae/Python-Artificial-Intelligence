# Training Guide

## Table of Contents
- [Training Guide](#training-guide)
  - [Table of Contents](#table-of-contents)
- [Regression](#regression)
  - [R² score](#r-score)
    - [Formula R² Score - Sklearn metrics](#formula-r-score---sklearn-metrics)
      - [R² = 1 - (SS\_res / SS\_tot)](#r--1---ss_res--ss_tot)
      - [Calcul](#calcul)
      - [R² = 1 - (SS\_res / SS\_tot)](#r--1---ss_res--ss_tot-1)
      - [Calcul](#calcul-1)
  - [MSE](#mse)
    - [Formula MSE - Sklearn metrics](#formula-mse---sklearn-metrics)
      - [MSE = (1/n) × Σ(y\_pred - y\_real)²](#mse--1n--σy_pred---y_real)
      - [Calcul](#calcul-2)
      - [Manual mse](#manual-mse)
  - [Mean Absolute Error](#mean-absolute-error)
    - [Formula MAE = Sklearn\_metrics](#formula-mae--sklearn_metrics)
      - [MAE = (1/n) \* Σ|y\_pred - y\_true|](#mae--1n--σy_pred---y_true)
      - [Calcul](#calcul-3)
  - [Efficiency Comparison](#efficiency-comparison)
  - [Best Choice Per Situation](#best-choice-per-situation)
    - [Choose MAE when:](#choose-mae-when)
    - [Choose MSE when:](#choose-mse-when)
  - [Industry Standards](#industry-standards)
  - [Regression Resume Example](#regression-resume-example)
- [Results Analysis](#results-analysis)
  - [Obtained Values](#obtained-values)
  - [Detailed Analysis](#detailed-analysis)
    - [1. R² = 0.84](#1-r--084)
    - [2. MSE = 26.2](#2-mse--262)
    - [3. MAE = 1.8](#3-mae--18)
  - [Conclusions](#conclusions)
    - [Positive Points:](#positive-points)
    - [Points of Attention:](#points-of-attention)
    - [Recommendations:](#recommendations)
- [Classification](#classification)
  - [ACCURACY](#accuracy)
    - [Formula accuracy - Sklearn metrics](#formula-accuracy---sklearn-metrics)
      - [Accuracy = (TP + TN) / (TP + TN + FP + FN)](#accuracy--tp--tn--tp--tn--fp--fn)
  - [Components](#components)
  - [Example](#example)
# Regression
## R² score
### Formula R² Score - Sklearn metrics
#### R² = 1 - (SS_res / SS_tot)


Example :
```python
y_real = [10,2,21,34,3]
y_pred = [11,43,7,4,1]
```
1. n = (n observations) 5
2. y_real = real values
3. y_pred = predicts values

#### Calcul
MSE = (1/5) * [
*  (11 - 10)² = 1² = 1 +
*  (43 - 2)² = 41² = 1681 +
*  (7 - 21)² = -14² = 196 +
*  (4 - 34)² = -30² = 900 +
*  (1 - 3)² = -2² = 4
  
]

MSE = (1/5) * 2782 = 556.4

 #### R² = 1 - (SS_res / SS_tot)

Example :
```python
y_real = [10,2,4,12,5]
y_pred = [11,6,11,4,4]
```
1. n = (n observations) 5
2. y_real = real values
3. y_pred = predicts values

#### Calcul
1. First calcule the mean
```python
    y_mean = (10+2+4+12+5) / 5 = 6.6
```
2. Calcul SS_res (Residual Sum of Square)
```python
    10 - 11 = (-1)²
    2 - 6 = (-4)²
    4 - 11 = (-7)² 
    12 - 4 = (8)²
    5 - 4 = (1)²

    SS_res = 1 + 16 + 49 + 64 + 1
    = 131
```
3. Calcule SS_tot
```python
    for each (n)y_true - mean
    10 - 6.6 = (3.4)² = 11.56
    2 - 6.6 = (-4.6)² = 21.16
    4 - 6.6 = (-2.6)² = 6.76
    12 - 6.6 = (5.4)² = 29.16
    5 - 6.6 = (-1.6)² = 2.56

    SS_tot = 71.2
```

4. Calculcate R²
```python
    R² = 1 - (SS_res / SS_tot)
    R² = 1 - (131 / 71.2)
    R² = 1 - 1.84
    R² = 0.84
```
## MSE
### Formula MSE - Sklearn metrics
 #### MSE = (1/n) × Σ(y_pred - y_real)²

Example :
```python
y_real = [10,2,4,12,5]
y_pred = [11,6,11,4,4]
```
1. n = (n observations) 5
2. y_real = real values
3. y_pred = predicts values

#### Calcul
MSE = (1/5) * [
*  (11 - 10)² = 1² = 1 +
*  (6 - 2)² = 4² = 16 +
*  (11 - 4)² = 7² = 49 +
*  (4 - 12)² = -8² = 64 +
*  (4 - 5)² = -1² = 1
  
]

MSE = (1/5) * 131 = 26.2

#### Manual mse

For this example `np.mean`  ==  (1/n) × Σ
```python
return np.mean((y_real - y_pred) ** 2)
```
## Mean Absolute Error
### Formula MAE = Sklearn_metrics
#### MAE = (1/n) * Σ|y_pred - y_true|
Example :
```python
y_real = [10,2,4,12,5]
y_pred = [11,6,11,4,4]
```
1. n = (n observations) 5
2. y_real = real values
3. y_pred = predicts values

#### Calcul
MAE = (1/5) * [
*  (11 - 10) = 1 | 1
* (6 - 2) = 4 | 4
* (11 - 4) = 7 | 7
* (4 - 12) = -8 | 8
* (4 - 5) = -1 | 1
]

MAE = (1/5) * (9) = 1.8

## Efficiency Comparison

| Criterion | MAE | MSE | Winner |
|-----------|-----|-----|---------|
| Speed of Computation | Faster (simple absolute) | Slower (squares needed) | MAE |
| Gradient Descent | Non-differentiable at 0 | Smooth and differentiable | MSE |
| Memory Usage | Less | More (larger numbers) | MAE |
| Outlier Handling | More stable | More sensitive | Depends on needs |

## Best Choice Per Situation

### Choose MAE when:
* Predicting financial values
* Working with time series
* Need interpretable results
* Have noisy data with outliers

### Choose MSE when:
* Training neural networks
* Need to penalize large errors
* Working on computer vision
* Need mathematical stability

## Industry Standards
* **Finance**: MAE (easier to explain dollars)
* **Deep Learning**: MSE (better for gradients)
* **Computer Vision**: MSE (better for pixel differences)
* **Time Series**: MAE (robust to outliers)

## Regression Resume Example

# Results Analysis

## Obtained Values
| Metric | Value | Interpretation |
|----------|---------|----------------|
| R² | 0.84 | 84% of variance explained |
| MSE | 26.2 | Mean Squared Error |
| MAE | 1.8 | Mean Absolute Error |

## Detailed Analysis

### 1. R² = 0.84
- **Excellent score** (close to 1)
- Model explains 84% of data variance
- Only 16% of variance remains unexplained
- Indicates very good overall predictive capability

### 2. MSE = 26.2
- High value due to squared errors
- Strongly influenced by large errors:
 * (11-4)² = 49
 * (4-12)² = 64

### 3. MAE = 1.8
- Average error of 1.8 units
- Easier to interpret than MSE
- Shows predictions are on average 1.8 units from actual values

## Conclusions

### Positive Points:
1. High R² (0.84) → Very good model
2. Relatively low MAE (1.8) → Accurate predictions

### Points of Attention:
1. High MSE (26.2) → Some large errors
2. Gap between MAE and MSE → Presence of outliers

### Recommendations:
1. Model is globally performing well (R² = 0.84)
2. Can be improved on specific predictions
3. Special attention needed for cases with large errors:
  - Prediction of 11 for actual value of 4
  - Prediction of 4 for actual value of 12
# Classification

## ACCURACY
Accuracy measures the proportion of correct predictions compared to the total number of predictions.
### Formula accuracy - Sklearn metrics
 #### Accuracy = (TP + TN) / (TP + TN + FP + FN)

Copy
## Components
- **TP (True Positive)**: Predicted 1, actual 1
- **TN (True Negative)**: Predicted 0, actual 0  
- **FP (False Positive)**: Predicted 1, actual 0
- **FN (False Negative)**: Predicted 0, actual 1

## Example
```python
y_pred = [0, 1, 0, 1, 0, 1, 0]
y_true = [0, 0, 1, 1, 1, 1, 0]

Analysis:
IndexPredictedActualResult000TN110FP201FN311
TP401FN511TP600TN

Count:

TP = 2 (positions 3, 5)
TN = 2 (positions 0, 6)
FP = 1 (position 1)
FN = 2 (positions 2, 4)

Calculation:
CopyAccuracy = (TP + TN) / Total
        = (2 + 2) / 7
        = 4/7
        ≈ 0.571 (57.1%)