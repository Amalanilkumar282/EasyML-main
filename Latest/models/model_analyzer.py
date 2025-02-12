import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from werkzeug.datastructures import FileStorage
import os
import json

def analyze_dataset(file_path, target=None):
    """
    Analyzes the dataset to determine the most suitable model type.
    Returns model type and target column suggestions.
    """
    try:
        # Read the dataset
        if isinstance(file_path, FileStorage):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_csv(file_path)
        
        # Calculate dataset characteristics
        dataset_stats = {
            'total_rows': len(df),
            'missing_values': df.isnull().sum().sum(),
            'numeric_features': df.select_dtypes(include=['int64', 'float64']).columns,
            'categorical_features': df.select_dtypes(include=['object']).columns
        }
        
        # If target is not provided, try to identify potential target columns
        if target is None:
            potential_targets = []
            for col in df.columns:
                unique_count = df[col].nunique()
                total_count = len(df)
                unique_ratio = unique_count / total_count
                
                # Enhanced target column criteria
                if unique_ratio < 0.7 and df[col].dtype in ['int64', 'float64', 'object']:
                    # Check for ID-like columns to exclude
                    if not (col.lower().endswith('id') or col.lower().startswith('id')):
                        potential_targets.append(col)
        else:
            if target not in df.columns:
                return {
                    "error": "Target column not found in dataset",
                    "status": "error"
                }
            potential_targets = [target]

        if not potential_targets:
            return {
                "error": "No suitable target columns found",
                "status": "error"
            }

        # Analyze each potential target
        results = []
        for target_col in potential_targets:
            target_data = df[target_col]
            
            # Enhanced feature analysis
            features = df.drop(columns=[target_col])
            
            # Check if target is numeric
            is_numeric = pd.api.types.is_numeric_dtype(target_data)
            
            # Check if target is categorical
            unique_values = target_data.nunique()
            is_categorical = unique_values < len(df) * 0.05 or not is_numeric
            
            # Check if target is binary
            is_binary = unique_values == 2
            
            # Calculate feature characteristics
            numeric_features = features.select_dtypes(include=['int64', 'float64'])
            categorical_features = features.select_dtypes(include=['object'])
            
            # Enhanced model selection criteria
            suitable_models = []
            
            # Multiple Linear Regression
            if is_numeric and not is_categorical:
                score = 0
                if len(numeric_features.columns) > 2:
                    score += 0.8
                    # Check for linear relationships
                    if numeric_features.shape[1] > 0:
                        correlations = numeric_features.corrwith(target_data).abs()
                        if correlations.mean() > 0.3:
                            score += 0.2
                    suitable_models.append(("multireg", score))
            
            # Simple Linear Regression
            if is_numeric and not is_categorical:
                score = 0
                if len(numeric_features.columns) <= 2:
                    score += 0.8
                    if numeric_features.shape[1] > 0:
                        correlations = numeric_features.corrwith(target_data).abs()
                        if correlations.mean() > 0.3:
                            score += 0.2
                    suitable_models.append(("reg", score))
            
            # Logistic Regression
            if is_categorical:
                score = 0
                if is_binary:
                    score += 0.8
                    if len(numeric_features.columns) > len(categorical_features.columns):
                        score += 0.2
                    suitable_models.append(("logreg", score))
            
            # Decision Tree
            if is_categorical:
                score = 0.6
                if len(df) < 1000:
                    score += 0.2
                if len(categorical_features.columns) > 0:
                    score += 0.2
                suitable_models.append(("dtree", score))
            
            # Naive Bayes
            if is_categorical:
                score = 0.5
                if is_binary:
                    score += 0.2
                if len(categorical_features.columns) > len(numeric_features.columns):
                    score += 0.2
                if len(df) < 1000:
                    score += 0.1
                suitable_models.append(("naivebayes", score))
            
            # KNN
            if is_categorical:
                score = 0.5
                if len(df) < 5000:
                    score += 0.2
                if len(numeric_features.columns) > len(categorical_features.columns):
                    score += 0.2
                if unique_values < 10:
                    score += 0.1
                suitable_models.append(("knn", score))
            
            # SVM
            if is_categorical:
                score = 0.5
                if is_binary:
                    score += 0.3
                if len(df) < 5000:
                    score += 0.1
                if len(numeric_features.columns) > len(categorical_features.columns):
                    score += 0.1
                suitable_models.append(("svm", score))
            
            # Sort models by score
            suitable_models.sort(key=lambda x: x[1], reverse=True)
            
            results.append({
                "target_column": target_col,
                "suitable_models": suitable_models,
                "is_numeric": is_numeric,
                "is_categorical": is_categorical,
                "unique_values": int(unique_values)
            })
        
        # Select the best result
        best_result = max(results, key=lambda x: x["suitable_models"][0][1] if x["suitable_models"] else 0)
        
        return {
            "status": "success",
            "best_model": best_result["suitable_models"][0][0] if best_result["suitable_models"] else None,
            "target_column": best_result["target_column"],
            "model_details": best_result
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "status": "error"
        }