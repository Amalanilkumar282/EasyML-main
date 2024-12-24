import pandas as pd
import numpy as np
import pickle
import os
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.svm import SVC
import matplotlib
matplotlib.use('agg')  # Use the 'agg' backend
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from scipy import stats
from config import MODEL_FOLDER  # Import the shared configuration



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
# def save_model(model, model_name='temp_model.pkl'):
#     """Save the trained model to a file using shared configuration"""
#     model_path = os.path.join(MODEL_FOLDER, model_name)
#     with open(model_path, 'wb') as f:
#         pickle.dump(model, f)
#     return model_path

def save_model(model, metrics, coefficients, p_values, intercept, plots):
    """
    Save the model and its associated data to a file
    """
    import os
    import pickle
    
    # Create a models directory if it doesn't exist
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Create a unique filename based on timestamp
    import time
    filename = f'model_{int(time.time())}.pkl'
    filepath = os.path.join(model_dir, filename)
    
    # Save all the model data
    model_data = {
        'model': model,
        'metrics': metrics,
        'coefficients': coefficients,
        'p_values': p_values,
        'intercept': intercept,
        'plots': plots
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    
    return filepath


def perform_multiple_linear_regression(file):
    # Read and prepare data
    df = pd.read_csv(file, index_col=False)
    
    # Get target column (last column)
    target = df.columns[-1]
    y = df[target].values
    
    # Remove target and ID column (if exists) from features
    drop_cols = [target]
    if 'id' in df.columns:
        drop_cols.append('id')
    X = df.drop(columns=drop_cols)
    feature_names = X.columns
    X = X.values
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    # Create plots list to store all plot images
    plots = []
    
    # 1. Feature scatter plots with regression lines
    n_features = X.shape[1]
    
    if n_features == 1:
        # Simple linear regression case
        plt.figure(figsize=(10, 6))
        plt.scatter(X_train, y_train, alpha=0.5, label='Training')
        plt.scatter(X_test, y_test, alpha=0.5, label='Testing')
        
        # Plot regression line
        x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_plot = model.predict(x_range)
        plt.plot(x_range, y_plot, color='red', label='Regression Line')
        
        plt.xlabel(str(feature_names[0]))
        plt.ylabel(target)
        plt.legend()
        plt.title('Simple Linear Regression')
        plots.append(get_plot_as_base64())
        
    else:
        # Multiple linear regression case
        n_cols = 2
        n_rows = (n_features + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        
        # Convert axes to array if single row
        if n_rows == 1:
            axes = np.array([axes])
            
        # Flatten axes for easier iteration
        axes = axes.ravel()
        
        # Hide any extra subplots
        if n_features < len(axes):
            for j in range(n_features, len(axes)):
                fig.delaxes(axes[j])
        
        for i, feature in enumerate(feature_names):
            ax = axes[i]
            
            ax.scatter(X_train[:, i], y_train, alpha=0.5, label='Training')
            ax.scatter(X_test[:, i], y_test, alpha=0.5, label='Testing')
            
            # Plot regression line
            x_range = np.linspace(X[:, i].min(), X[:, i].max(), 100)
            X_plot = np.zeros((100, X.shape[1]))
            X_plot[:, i] = x_range
            y_plot = model.predict(X_plot)
            ax.plot(x_range, y_plot, color='red', label='Regression Line')
            
            ax.set_xlabel(str(feature))
            ax.set_ylabel(target)
            ax.legend()
        
        plt.tight_layout()
        plots.append(get_plot_as_base64())
    
    # 2. Correlation heatmap
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    plots.append(get_plot_as_base64())
    
    # 3. Residual plot
    plt.figure(figsize=(10, 6))
    residuals = y_test - y_pred_test
    plt.scatter(y_pred_test, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plots.append(get_plot_as_base64())
    
    # 4. Actual vs Predicted plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_test)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted')
    plots.append(get_plot_as_base64())
    
    # Calculate feature importance and p-values
    n = X_train.shape[0]
    p = X_train.shape[1]
    dof = n - p - 1
    mse = np.sum((y_test - y_pred_test) ** 2) / dof
    var_b = mse * np.linalg.inv(np.dot(X_train.T, X_train)).diagonal()
    sd_b = np.sqrt(var_b)
    t_stat = model.coef_ / sd_b
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stat), dof))
    
    # Create dictionaries for coefficients and p-values
    coefficients = dict(zip(feature_names, model.coef_))
    p_values_dict = dict(zip(feature_names, p_values))
    
    model_path = save_model(
        model=model,
        metrics={
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse
        },
        coefficients=coefficients,
        p_values=p_values_dict,
        intercept=model.intercept_,
        plots=plots
    )
    
    return (
        model_path,
        plots,
        {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse
        },
        coefficients,
        p_values_dict,
        model.intercept_
    )



def get_plot_as_base64():
    image_stream = BytesIO()
    plt.savefig(image_stream, format='png', bbox_inches='tight')
    image_stream.seek(0)
    img_str = base64.b64encode(image_stream.read()).decode('utf-8')
    plt.close()
    return img_str


def predict_new_values(model, feature_names, values_dict):
    """
    Make predictions using the trained multiple linear regression model
    
    Parameters:
    model: trained LinearRegression model
    feature_names: list of feature names
    values_dict: dictionary containing feature values
    
    Returns:
    prediction: numpy array containing the predicted value
    """
    import numpy as np
    import pandas as pd
    
    # Create a DataFrame with a single row containing the new values
    new_data = pd.DataFrame([values_dict])
    
    # Ensure the features are in the correct order
    new_data = new_data[feature_names]
    
    # Make prediction
    prediction = model.predict(new_data)
    
    return prediction


##############################################################################################
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