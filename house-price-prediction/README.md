# House Price Prediction
Linear regression model to predict house prices based on property features and characteristics.

## Overview
This project implements a comprehensive house price prediction system using linear regression. The model analyzes various property features like bedrooms, bathrooms, square footage, and location factors to predict market prices.

## Features
- Exploratory data analysis with correlation matrix
- Data visualization using pairplots and heatmaps
- Missing value detection and handling
- Linear regression model implementation
- Model performance evaluation with residual analysis
- Feature importance analysis

## Technologies Used
- Python 3.x
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- Linear regression algorithm

## Dataset
Expected CSV file with columns:
- `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`
- `floors`, `waterfront`, `view`, `condition`
- `price` (target variable)

## Setup
1. Ensure data file is located at `data/data.csv`
2. Install requirements: `pip install pandas numpy scikit-learn matplotlib seaborn`
3. Run the script to train model and view results

## Model Performance
- R² score for model accuracy assessment
- Coefficient analysis for feature importance
- Residual plots for prediction quality validation
- Correlation analysis for feature relationships

## Learning Objectives
- Linear regression implementation and evaluation
- Data preprocessing and visualization techniques
- Model performance metrics interpretation
- Feature correlation analysis
- Residual analysis for model validation