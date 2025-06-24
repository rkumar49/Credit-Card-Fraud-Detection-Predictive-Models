# Credit Card Fraud Detection System

## Overview
This project develops a machine learning system to detect fraudulent credit card transactions using a highly imbalanced dataset (492 frauds in 284,807 transactions). The solution employs advanced feature analysis and multiple classification algorithms to identify fraud patterns while handling significant data imbalance challenges.

## Key Steps

### 1. Data Exploration & Preprocessing
- Analyzed transaction distribution over time
- Visualized class imbalance (0.172% fraud rate)
- Examined feature distributions and correlations
- Engineered temporal features (transaction hour)
- Verified no missing values

### 2. Feature Analysis
- Identified key fraud indicators through:
  - Density plots of PCA-transformed features
  - Correlation heatmaps
  - Transaction pattern analysis by hour
- Discovered critical features: V17, V14, V12, V10, V11

### 3. Modeling Approach
- Implemented 4 classification models:
  - Random Forest (baseline)
  - AdaBoost
  - CatBoost
  - XGBoost (best performer)
- Addressed class imbalance through:
  - AUC-ROC optimization
  - Class-weighted evaluation
  - Feature importance analysis

### 4. Validation Strategy
- 3-way data split: Train (64%), Validation (16%), Test (20%)
- K-Fold cross-validation (K=5)
- Early stopping to prevent overfitting
- Strict separation of test data

### 5. Performance
- Achieved **0.977 AUC-ROC** on test data using XGBoost
- Confusion matrix analysis for Type I/II error tradeoffs
- Feature importance visualization

## Technical Stack
- Python, Pandas, NumPy
- Scikit-learn, XGBoost, CatBoost
- Matplotlib, Seaborn, Plotly
- Jupyter Notebook
