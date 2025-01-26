import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from flask import render_template, jsonify, request, redirect, url_for
from models.regression import (
    perform_linear_regression, 
    perform_multiple_linear_regression
)

from models.classification import (
    perform_logistic_regression,
    perform_knn,
    perform_dtree,
    perform_naivebayes,
    perform_svm
)

def analyze_dataset(file_path):
    """
    Analyze the uploaded dataset and determine the most suitable model
    
    Returns:
    - Model type (regression/classification)
    - Best model name
    - Target column (if applicable)
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Check for target column determination
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    # Determine model type and potential target column
    def is_continuous_target(series):
        return len(series.unique()) > 10 and series.dtype in ['float64', 'int64']
    
    potential_targets = [col for col in numeric_columns if is_continuous_target(df[col])]
    
    # Classification indicators
    classification_indicators = len(categorical_columns) > 0
    
    # Model selection logic
    if len(potential_targets) > 0:
        # Regression scenario
        best_model, metrics = select_best_regression_model(df, potential_targets[0])
        return 'regression', best_model, potential_targets[0]
    elif classification_indicators:
        # Classification scenario
        categorical_target = categorical_columns[0] if len(categorical_columns) > 0 else None
        best_model, metrics = select_best_classification_model(df, categorical_target)
        return 'classification', best_model, categorical_target
    else:
        return None, None, None

def select_best_regression_model(df, target):
    """Select best regression model based on performance metrics"""
    models = {
        'Linear Regression': perform_linear_regression,
        'Multiple Linear Regression': perform_multiple_linear_regression,
        'KNN Regression': perform_knn,
        'Decision Tree Regression': perform_dtree,
        'SVM Regression': perform_svm
    }
    
    model_metrics = {}
    
    for name, model_func in models.items():
        try:
            # Print debug information
            print(f"Attempting to train {name}")
            
            # Perform model training and get metrics
            metrics = model_func(df, target)
            
            # Check if metrics are valid
            if metrics and 'r2_score' in metrics:
                model_metrics[name] = metrics['r2_score']
                print(f"{name} R² Score: {metrics['r2_score']}")
            else:
                print(f"Invalid metrics for {name}")
        except Exception as e:
            print(f"Error in {name}: {str(e)}")
            continue
    
    # Check if any models were successfully trained
    if not model_metrics:
        raise ValueError("No models could be trained successfully. Check your dataset and model functions.")
    
    # Select best model based on highest R² score
    best_model = max(model_metrics, key=model_metrics.get)
    return best_model, model_metrics

def select_best_classification_model(df, target):
    """Select best classification model based on performance metrics"""
    models = {
        'Logistic Regression': perform_logistic_regression,
        'KNN': perform_knn,
        'Decision Tree': perform_dtree,
        'Naive Bayes': perform_naivebayes,
        'SVM': perform_svm
    }
    
    model_metrics = {}
    
    for name, model_func in models.items():
        try:
            # Print debug information
            print(f"Attempting to train {name}")
            
            # Perform model training and get metrics
            metrics = model_func(df, target)
            
            # Check if metrics are valid
            if metrics and 'accuracy' in metrics:
                model_metrics[name] = metrics['accuracy']
                print(f"{name} Accuracy: {metrics['accuracy']}")
            else:
                print(f"Invalid metrics for {name}")
        except Exception as e:
            print(f"Error in {name}: {str(e)}")
            continue
    
    # Check if any models were successfully trained
    if not model_metrics:
        raise ValueError("No models could be trained successfully. Check your dataset and model functions.")
    
    # Select best model based on highest accuracy
    best_model = max(model_metrics, key=model_metrics.get)
    return best_model, model_metrics

