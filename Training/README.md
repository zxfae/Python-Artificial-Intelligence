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
      - [Manual mse](#manual-mse)
  - [MSE](#mse)
    - [Formula MSE - Sklearn metrics](#formula-mse---sklearn-metrics)
      - [MSE = (1/n) × Σ(y\_pred - y\_real)²](#mse--1n--σy_pred---y_real)
      - [Calcul](#calcul-2)
      - [Manual mse](#manual-mse-1)
  - [Mean Absolute Error](#mean-absolute-error)
    - [Formula MAE = Sklearn\_metrics](#formula-mae--sklearn_metrics)
      - [MAE = (1/n) \* Σ|y\_pred - y\_true|](#mae--1n--σy_pred---y_true)
      - [Calcul](#calcul-3)
  - [Efficiency Comparison](#efficiency-comparison)
  - [Best Choice Per Situation](#best-choice-per-situation)
    - [Choose MAE when:](#choose-mae-when)
    - [Choose MSE when:](#choose-mse-when)
  - [Industry Standards](#industry-standards)
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

#### Manual mse

For this example `np.mean`  ==  (1/n) × Σ
```python
return np.mean((y_real - y_pred) ** 2)
```
## MSE
### Formula MSE - Sklearn metrics
 #### MSE = (1/n) × Σ(y_pred - y_real)²

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
MSE = (1/5) * [
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