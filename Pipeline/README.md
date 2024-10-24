# Understanding Pipeline

## Table of Contents
- [Understanding Pipeline](#understanding-pipeline)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [What is Pipeline?](#what-is-pipeline)
  - [SimpleImputer](#simpleimputer)
    - [Using (Step by Step):](#using-step-by-step)
  - [Scaler](#scaler)
    - [Using (Step by Step):](#using-step-by-step-1)
      - [Formula](#formula)
    - [Pipeline](#pipeline)
      - [Strategy](#strategy)
        - [Classifier](#classifier)
      - [1. General Characteristics](#1-general-characteristics)
      - [2. Performance Metrics](#2-performance-metrics)
      - [3. Implementation Details](#3-implementation-details)
      - [4. Data Compatibilty](#4-data-compatibilty)
  - [Train-Test Split](#train-test-split)
  - [Model Evaluation Metrics](#model-evaluation-metrics)
  - [Pipeline Operations Understanding](#pipeline-operations-understanding)
  - [Cross-Validation](#cross-validation)
  - [Model Persistence](#model-persistence)
  - [Handling Different Data Types](#handling-different-data-types)
  - [Common Warnings and Best Practices](#common-warnings-and-best-practices)
  - [Quick Tips for Pipeline Usage](#quick-tips-for-pipeline-usage)
  - [Common Pipeline Patterns](#common-pipeline-patterns)

## Introduction
Pipeline is a crucial tool in machine learning that allows you to chain multiple steps together, ensuring a consistent workflow between data preprocessing and model training. It helps prevent data leakage and simplifies the process of applying the same transformations to both training and test datasets.

## What is Pipeline?
A Pipeline is a sequence of data processing components that are executed in a specific order. Each component in the pipeline takes data as input, processes it, and passes the transformed data to the next component. This is particularly useful for:
- Ensuring preprocessing steps are applied in the same order
- Preventing data leakage between training and test sets
- Simplifying the machine learning workflow
- Making code more maintainable and less prone to errors

## SimpleImputer
SimpleImputer is used to handle missing values in datasets by replacing them with calculated statistics (like mean, median) or constants.

### Using (Step by Step):
1. Creation of imputer
```
imp_mean = SimpleImputer(missing_value=np.nan, stragery='mean')
```
2. Calculate mean with fit
```
.fit(datas)
```
3. Apply transformation
```
datas_imputed = imp_mean.transform(datas)
print(datas_imputed)
```

## Scaler
Don't use it when:

- With Decision Trees algorithms
- With Random Forests algorithms
- When features are already on the same scale (example: grades from 0 to 20)...

### Using (Step by Step):
#### Formula
``` 
z = (x - mean) / standardDeviation(std)
```
1. Initialize the Standard Scaler
```
scaler = StandardScaler()
```
2. fit_transform the datas_X
```
scaler.fit_transform(datas_X)
```
3. Apply transformation
```
datas_imputed = imp_mean.transform(datas)
print(datas_imputed)
```

### Pipeline
You can use the Pipeline method provide by SciKit-Learn

```
from sklearn.pipeline import Pipeline
```

Example  With Pipeline method and parameters:

```
pipeline = Pipeline([
  ('imputer', SimpleImputer(strategy='median')),
  ('scaler', StandardScaler()),
  ('classifier', LogisticRegression(max_iter=n))
])
```
#### Strategy

 by default's mean(replace missing values by the mean along each column), but we have :

* median : replace missing value by the median along each column
* most_frequent : then replace missing using the most frequent value along each column
* constant : replace missing values with fill_value
* callable : replace missing values using the scalar statistic

##### Classifier
#### 1. General Characteristics
| Algorithm | Advantages | Disadvantages | Best Use Cases |
|:---:|:---:|:---:|:---|
| Logistic Regression | Fast, Simple, Interpretable | Only linear boundaries | Binary classification, Simple datasets with clear separation |
| SVM | Powerful, Works in high dimensions | Slow on large datasets, Complex tuning | Image classification, Complex non-linear problems |
| KNN | No training, Works with any distribution | Slow predictions, Memory intensive | Small datasets, Recommendation systems |
| Decision Tree | Very interpretable, No scaling needed | Can overfit, Unstable | Categorical data, Financial analysis |
| Gradient Boosting | Best accuracy, Handles mixed data | Slow training, Complex tuning | Competitions, When accuracy is key |

#### 2. Performance Metrics
| Algorithm | Training Speed | Memory Usage | Required Preprocessing |
|:---:|:---:|:---:|:---|
| Logistic Regression | Fast ⚡ | Low 💾 | Scaling, Handle missing values |
| SVM | Slow 🐢 | Medium 💾💾 | Mandatory scaling, Handle missing values |
| KNN | None ✨ | High 💾💾💾 | Scaling, Handle missing values |
| Decision Tree | Fast ⚡ | Low 💾 | Only handle missing values |
| Gradient Boosting | Slow 🐢 | Medium 💾💾 | Only handle missing values |

#### 3. Implementation Details
| Algorithm | Complexity | Key Parameters | Typical Values |
|:---:|:---:|:---:|:---|
| Logistic Regression | Low | C | 0.1, 1.0, 10.0 |
| SVM | High | kernel, C | 'linear'/'rbf', C: 0.1-10 |
| KNN | Medium | n_neighbors | 3, 5, 7, 11 |
| Decision Tree | Medium | max_depth | 3 to 10 |
| Gradient Boosting | Very High | n_estimators, learning_rate | 100-1000, 0.01-0.1 |

#### 4. Data Compatibilty
| Algorithm | Dataset Size | Feature Types | Interpretability |
|:---:|:---:|:---:|:---|
| Logistic Regression | Large ✓ | Numerical | High 🔍 |
| SVM | Small/Medium | Numerical | Low 🔍 |
| KNN | Small | All types | Medium 🔍 |
| Decision Tree | All sizes | All types | Very High 🔍 |
| Gradient Boosting | All sizes | All types | Low 🔍 |

## Train-Test Split
Before creating our pipeline, we need to split our data into training and testing sets:
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.33,    # 33% for testing
    random_state=42,   # For reproducibility
    stratify=y         # For balanced class distribution
)
```

## Model Evaluation Metrics

| Metric | Description | When to Use |
|:---:|:---:|:---|
| Accuracy | Correct predictions / Total predictions | Balanced datasets |
| Precision | True Positives / (True + False Positives) | Cost of false positives is high |
| Recall | True Positives / (True + False Negatives) | Cost of false negatives is high |
| F1-Score | 2 * (Precision * Recall) / (Precision + Recall) | Need balance of precision/recall |
| ROC-AUC | Area under ROC curve | Binary classification evaluation |

## Pipeline Operations Understanding

| Operation | Training Data | Test Data | Purpose |
|:---:|:---:|:---:|:---|
| fit() | ✓ | ✗ | Learn parameters from data |
| transform() | ✓ | ✓ | Apply learned parameters |
| fit_transform() | ✓ | ✗ | Learn and apply in one step |
| predict() | ✓ | ✓ | Make predictions using learned parameters |

## Cross-Validation
Cross-validation helps assess model performance more robustly:

```python
# Basic cross-validation
from sklearn.model_selection import cross_val_score

scores = cross_val_score(pipeline, X, y, cv=5)
print(f"CV Scores: {scores}")
print(f"Average CV Score: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

# Grid Search with cross-validation
from sklearn.model_selection import GridSearchCV

param_grid = {
    'classifier__C': [0.1, 1.0, 10.0],
    'classifier__max_iter': [100, 200, 300]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X_train, y_train)
```

## Model Persistence
Save and load your trained pipeline:

```python
from joblib import dump, load

# Save the model
dump(pipeline, 'model_pipeline.joblib')

# Load the model
loaded_pipeline = load('model_pipeline.joblib')
```

## Handling Different Data Types
When dealing with mixed data types:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Define feature types
numeric_features = ['age', 'salary']
categorical_features = ['department', 'position']

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Use in pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])
```

## Common Warnings and Best Practices

✅ DO:
- Always split data before preprocessing
- Use pipeline to prevent data leakage
- Save the entire pipeline, not just the model
- Use cross-validation for robust evaluation

❌ DON'T:
- Preprocess data before splitting
- Fit preprocessors on test data
- Share parameters between train and test sets
- Forget to handle categorical variables

## Quick Tips for Pipeline Usage

1. **Accessing Steps**:
```python
# Get a specific step
scaler = pipeline.named_steps['scaler']

# Get parameters
params = pipeline.get_params()
```

2. **Debugging**:
```python
# Set verbose parameter
pipeline = Pipeline([...], verbose=True)

# Check intermediate outputs
pipeline.named_steps['scaler'].transform(X)
```

3. **Memory Efficiency**:
```python
# Cache transformations
from tempfile import mkdtemp
from sklearn.pipeline import Pipeline
from sklearn.externals.joblib import Memory

cachedir = mkdtemp()
memory = Memory(cachedir=cachedir, verbose=0)

pipeline = Pipeline([...], memory=memory)
```

## Common Pipeline Patterns

1. **Basic Classification**:
```python
pipeline = Pipeline([
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])
```

2. **Feature Selection**:
```python
pipeline = Pipeline([
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(k=10)),
    ('classifier', LogisticRegression())
])
```

3. **Dimensionality Reduction**:
```python
pipeline = Pipeline([
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),
    ('classifier', LogisticRegression())
])
```