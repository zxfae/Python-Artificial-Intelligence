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
  - [Precision](#precision)
    - [Formula Precisions - Sklearn metrics](#formula-precisions---sklearn-metrics)
      - [Precision = (TP) / (TP + FP)](#precision--tp--tp--fp)
  - [Components](#components-1)
  - [Example](#example-1)
  - [Recall](#recall)
    - [Formula Recall - Sklearn metrics](#formula-recall---sklearn-metrics)
      - [Precision = (TP) / (TP + FN)](#precision--tp--tp--fn)
  - [Components](#components-2)
  - [Example](#example-2)
  - [F1\_score](#f1_score)
    - [Formula F1\_score - Sklearn metrics](#formula-f1_score---sklearn-metrics)
      - [Precision = 2 \* (Precision \* Recall) / (Precision + Recall)](#precision--2--precision--recall--precision--recall)
  - [Components](#components-3)
  - [Example](#example-3)
  - [ROC AUC](#roc-auc)
    - [Formula ROC AUC - Sklearn metrics](#formula-roc-auc---sklearn-metrics)
      - [AUC = ∫ TPR d(FPR)](#auc---tpr-dfpr)
  - [Components](#components-4)
  - [Example](#example-4)
  - [Interpretation](#interpretation)
- [Classification Metrics Guide](#classification-metrics-guide)
  - [Metrics Comparison Table](#metrics-comparison-table)
  - [Quick Selection Guide](#quick-selection-guide)
    - [Choose based on context:](#choose-based-on-context)
    - [Practical Rules:](#practical-rules)
    - [Common Values Interpretation:](#common-values-interpretation)
    - [Selection Formula:](#selection-formula)
# Regression
## R² score
### Formula R² Score - Sklearn metrics
#### R² = 1 - (SS_res / SS_tot)
#### Calcul
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

MAE = (1/5) * (21) = 4.2

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
```

## Precision
### Formula Precisions - Sklearn metrics
 #### Precision = (TP) / (TP + FP)

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
CopyPrecision = (TP) / (TP + FP)
        = (2) / 3
        ≈ 0.66 (66%)
```
## Recall
### Formula Recall - Sklearn metrics
 #### Precision = (TP) / (TP + FN)

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
CopyRecall = (TP) / (TP + FN)
        = (2) / 4
        ≈ 0.5 (50%)
```

## F1_score
### Formula F1_score - Sklearn metrics
 #### Precision = 2 * (Precision * Recall) / (Precision + Recall)

Copy
## Components
- **TP (True Positive)**: Predicted 1, actual 1
- **TN (True Negative)**: Predicted 0, actual 0  
- **FP (False Positive)**: Predicted 1, actual 0
- **FN (False Negative)**: Predicted 0, actual 1

## Example
```python
Calc = 2 * (0.66 * 0.5) / (0.66 + 0.5)
Calc = 0.569(56,9%)
```
## ROC AUC
### Formula ROC AUC - Sklearn metrics

#### AUC = ∫ TPR d(FPR)
- TPR (True Positive Rate) = TP / (TP + FN)
- FPR (False Positive Rate) = FP / (FP + TN)

## Components
- **TPR (True Positive Rate)**:  Recall
- **FPR (False Positive Rate)**: Also called 1 - Specificity
- **Thresholds**: Values between 0 and 1 used to convert probabilities to classes
- **Area Under Curve**: Integral of the ROC curve (area between curve and x-axis)


## Example
```python
fpr = [0, 0.2, 0.4, 0.6, 0.8, 1]  
tpr = [0, 0.6, 0.8, 0.9, 0.95, 1] 

auc = 0
for i in range(len(fpr)-1):
    width = fpr[i+1] - fpr[i]
    height = (tpr[i+1] + tpr[i]) / 2
    auc += width * height

Result: AUC = 0.85 (85%)
```

## Interpretation 
1. AUC = 1.0: Perfect classification
2. AUC = 0.5: Random classification
3. AUC > 0.9: Excellent
4. AUC > 0.8: Good
5. AUC > 0.7: Acceptable
6. AUC < 0.6: Poor

# Classification Metrics Guide

## Metrics Comparison Table

| Metric | When to Use | Advantages | Disadvantages | Use Cases |
|--------|-------------|------------|---------------|------------|
| **Accuracy** | - Balanced classes<br>- Equal error costs | - Easy to understand<br>- Intuitive | - Misleading with imbalanced classes | - Image classification<br>- Balanced spam detection |
| **Precision** | - Minimize false positives<br>- High FP cost | - Good for avoiding false alarms<br>- Measures positive prediction quality | - Ignores FN<br>- Can be biased | - Medical diagnosis<br>- Fraud detection |
| **Recall** | - Minimize false negatives<br>- High FN cost | - Finds all positive cases<br>- Sensitive to rare cases | - Ignores FP<br>- Can generate too many alerts | - Disease detection<br>- Critical defect detection |
| **F1-Score** | - Imbalanced classes<br>- Balance precision/recall | - Combines precision and recall<br>- Good for imbalance | - Ignores TN<br>- Can mask details | - NLP<br>- Recommendation systems |
| **ROC AUC** | - Evaluate overall performance<br>- Compare models | - Threshold independent<br>- Robust to imbalance | - Can be too optimistic<br>- Complex to interpret | - Credit scoring<br>- Model ranking |

## Quick Selection Guide

### Choose based on context:

1. **Balanced data & simplicity** 
  - ➡️ Accuracy
  - Example: Simple classifications

2. **High cost of false positives**
  - ➡️ Precision
  - Example: Medical diagnosis (avoid unnecessary treatments)

3. **High cost of false negatives**
  - ➡️ Recall
  - Example: Cancer detection (don't miss cases)

4. **Imbalanced data**
  - ➡️ F1-Score or ROC AUC
  - Example: Fraud detection (few positive cases)

5. **Model comparison**
  - ➡️ ROC AUC
  - Example: Best model selection

### Practical Rules:

- **Medical/Health**: Favor Recall
 - Critical to not miss positive cases
 - False negatives are very costly

- **Finance/Fraud**: Balance Precision/Recall (F1)
 - Need to catch fraud without too many false alarms
 - Both types of errors are costly

- **Marketing**: ROC AUC for scoring
 - Good for ranking and probability estimates
 - Compare different models' performance

- **Industrial Production**: Precision
 - Quality control
 - Minimize false alarms

### Common Values Interpretation:

| Metric | Poor | Acceptable | Good | Excellent |
|--------|------|------------|------|-----------|
| Accuracy | < 0.6 | 0.6 - 0.7 | 0.7 - 0.8 | > 0.8 |
| Precision | < 0.5 | 0.5 - 0.7 | 0.7 - 0.9 | > 0.9 |
| Recall | < 0.5 | 0.5 - 0.7 | 0.7 - 0.9 | > 0.9 |
| F1-Score | < 0.5 | 0.5 - 0.7 | 0.7 - 0.8 | > 0.8 |
| ROC AUC | < 0.6 | 0.6 - 0.7 | 0.7 - 0.8 | > 0.8 |

### Selection Formula:

1. **If balanced_classes and simple_problem**:
  - Use Accuracy

2. **If imbalanced_classes**:
  ```python
  if cost_of_false_positives > cost_of_false_negatives:
      use_precision()
  elif cost_of_false_negatives > cost_of_false_positives:
      use_recall()
  else:
      use_f1_score()