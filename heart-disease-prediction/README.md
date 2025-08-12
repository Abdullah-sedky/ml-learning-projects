# Heart Disease Prediction
Logistic regression model for binary classification to predict heart disease risk based on medical indicators.

## Overview
This project implements a heart disease prediction system using logistic regression for binary classification. The model analyzes various medical features and patient characteristics to predict the likelihood of heart disease occurrence.

## Features
- Binary classification using logistic regression
- Correlation analysis with heatmap visualization
- Missing value detection and handling
- Probability predictions and classification
- Comprehensive model evaluation metrics
- Confusion matrix analysis

## Technologies Used
- Python 3.x
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- Logistic regression algorithm

## Dataset
Expected CSV file (`data/heart-2.csv`) with medical features and:
- Multiple health indicator columns
- `target` column (binary: 0=no heart disease, 1=heart disease)

## Setup
1. Ensure data file is located at `data/heart-2.csv`
2. Install requirements: `pip install pandas numpy scikit-learn matplotlib seaborn`
3. Run the script to train model and view evaluation metrics

## Model Performance
- Accuracy score for overall prediction correctness
- Log loss for probability prediction quality
- ROC-AUC score for classification performance
- Confusion matrix for detailed prediction analysis
- F1 score for balanced precision-recall metric
- Feature coefficients for importance analysis

## Learning Objectives
- Logistic regression for binary classification
- Medical data preprocessing and analysis
- Classification evaluation metrics interpretation
- Probability vs binary prediction understanding
- Missing data handling techniques