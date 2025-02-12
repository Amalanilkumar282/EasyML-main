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
        
        # If target is not provided, try to identify potential target columns
        if target is None:
            potential_targets = []
            for col in df.columns:
                unique_count = df[col].nunique()
                total_count = len(df)
                unique_ratio = unique_count / total_count
                
                # Potential target columns criteria
                if unique_ratio < 0.5 and df[col].dtype in ['int64', 'float64', 'object']:
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
            
            # Check if target is numeric
            is_numeric = pd.api.types.is_numeric_dtype(target_data)
            
            # Check if target is categorical
            unique_values = target_data.nunique()
            is_categorical = unique_values < len(df) * 0.05 or not is_numeric
            
            # Check if target is binary
            is_binary = unique_values == 2
            
            # Check if features are primarily numeric
            numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
            numeric_ratio = len(numeric_features) / len(df.columns)
            
            # Determine suitable models
            suitable_models = []
            
            if is_numeric and not is_categorical:
                if len(numeric_features) > 2:
                    suitable_models.append(("multireg", 0))
                else:
                    suitable_models.append(("reg", 0))
            
            if is_categorical:
                if is_binary:
                    suitable_models.append(("logreg", 0))
                suitable_models.append(("dtree", 0))
                suitable_models.append(("naivebayes", 0))
                suitable_models.append(("knn", 0))
                suitable_models.append(("svm", 0))
            
            # Score models based on dataset characteristics
            for i, (model, score) in enumerate(suitable_models):
                # Add score based on numeric feature ratio
                if model in ["multireg", "reg", "logreg", "svm"]:
                    suitable_models[i] = (model, score + numeric_ratio)
                
                # Add score based on dataset size
                if len(df) > 1000 and model in ["svm", "knn"]:
                    suitable_models[i] = (model, score - 0.2)
                elif len(df) < 1000 and model in ["dtree", "naivebayes"]:
                    suitable_models[i] = (model, score + 0.2)
                
                # Add score based on number of classes
                if is_categorical and not is_binary and model == "logreg":
                    suitable_models[i] = (model, score - 0.5)
            
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
