import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.svm import SVC
import matplotlib
matplotlib.use('agg')  # Use the 'agg' backend
import matplotlib.pyplot as plt
from io import BytesIO
import base64

#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################

def perform_linear_regression(file):
    df = pd.read_csv(file, index_col=False)
    target = df.columns[-1]
    y = df[target].values.reshape(-1, 1)
    target = [target]
    if 'id' in df.columns:
        target.append('id')
    X = df.drop(columns=target).values.reshape(-1, 1)

    # Perform Linear Regression
    model = LinearRegression()
    model.fit(X, y)

    # Create scatter plot and regression line
    plt.scatter(X, y)
    plt.plot(X, model.predict(X), color='red')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # Save plot to BytesIO object
    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    image_stream.seek(0)
    img_str = base64.b64encode(image_stream.read()).decode('utf-8')

    plt.close()

    return model, img_str

#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################

def perform_multiple_linear_regression(file):
    df = pd.read_csv(file, index_col=False)
    
    # Get target column (last column)
    target = df.columns[-1]
    y = df[target].values
    
    # Remove target and ID column (if exists) from features
    drop_cols = [target]
    if 'id' in df.columns:
        drop_cols.append('id')
    X = df.drop(columns=drop_cols).values
    
    # Perform Multiple Linear Regression
    model = LinearRegression()
    model.fit(X, y)
    
    # Create scatter plots for each feature
    n_features = X.shape[1]
    fig, axes = plt.subplots(1, n_features, figsize=(5*n_features, 4))
    
    # Handle case where there's only one feature
    if n_features == 1:
        axes = [axes]
    
    feature_names = df.drop(columns=drop_cols).columns
    
    for i, ax in enumerate(axes):
        ax.scatter(X[:, i], y)
        
        # Plot regression line for this feature
        sorted_idx = np.argsort(X[:, i])
        ax.plot(X[sorted_idx, i], model.predict(X)[sorted_idx], color='red')
        
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel(target)
    
    plt.tight_layout()
    
    # Save plot to BytesIO object
    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    image_stream.seek(0)
    img_str = base64.b64encode(image_stream.read()).decode('utf-8')
    
    plt.close()
    
    # Calculate R-squared score
    r2_score = model.score(X, y)
    
    # Get feature coefficients
    coefficients = dict(zip(feature_names, model.coef_))
    intercept = model.intercept_
    
    return model, img_str, r2_score, coefficients, intercept

#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################

def perform_svm_reg(file,target):
    df = pd.read_csv(file,index_col=False)
    y=df[target]
    target=[target]
    if 'id' in df.columns:
        target.append('id')
    X=df.drop(columns=target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)
    model=SVC(kernel='linear')
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)

    conf_matrix = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    image_stream.seek(0)
    img_str = base64.b64encode(image_stream.read()).decode('utf-8')
    return(accuracy,img_str,conf_matrix)

#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################