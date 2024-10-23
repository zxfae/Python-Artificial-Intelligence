# Models Guide

## Table of Contents
- [Models Guide](#models-guide)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Random forest](#random-forest)
  - [Key Scenarios and Advantages](#key-scenarios-and-advantages)
  - [Technical Details](#technical-details)

## Introduction
This guide provides an overview of linear regression, its components, and methods for evaluating model performance.

## Random forest

## Key Scenarios and Advantages

| Scenario | Why Random Forest? | Example |
|----------|-------------------|---------|
| Imbalanced Data | Naturally handles class imbalance | Credit card fraud detection (99% normal vs 1% fraudulent) |  
| Missing Values | Can process datasets with NaN values | Customer data with incomplete fields |
| Mixed Features | Works with numerical/categorical without transformation | Housing data (price, location, type) |
| Large Datasets | Parallelizable and scalable | Millions of user interactions |
| Feature Selection | Built-in feature importance | Identifying key factors in medical diagnosis |

## Technical Details

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

# Basic implementation
rf = RandomForestClassifier(
   n_estimators=100,     
   # Number of trees
   max_depth=None,       
   # Maximum tree depth 
   min_samples_split=2,  
   # Minimum samples to split
   random_state=42       
   # For reproducibility
)
```
| Model | Best Performance | Worst Performance |
|-------|-----------------|-------------------|
| Linear Regression | Linear data | Non-linear data |
| SVM | High dimensions | Large datasets |
| Decision Tree | Categorical data | Continuous data |
| Random Forest | Complex data | Simple linear data |
| Gradient Boosting | Mixed data types | Small datasets |


| Type | Description | Exemple | Taille Approximative |
|------|-------------|---------|---------------------|
| Non-Linear Data | Relations qui ne suivent pas une ligne droite | y = x² ou y = sin(x) | N/A |
| Linear Data | Relations qui suivent une ligne droite | y = 2x + 1 | N/A |
| Large Dataset | Données dépassant la mémoire RAM standard | >100,000 lignes | >1GB |
| Small Dataset | Données tenant facilement en mémoire | <1,000 lignes | <10MB |
| Continuous Data | Données numériques avec valeurs infinies | Température, Prix | N/A |