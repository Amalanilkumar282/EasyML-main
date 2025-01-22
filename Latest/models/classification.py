import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib
matplotlib.use('agg')  # Use the 'agg' backend
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import seaborn as sns
import os
import numpy as np
from werkzeug.datastructures import FileStorage
#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################
def perform_analysis(file,target):
    temp_folder = 'temp_anal'
    file_path = os.path.join(temp_folder, 'f')   
    file.save(file_path)
    file=FileStorage(filename='f', stream=open('temp_anal/f', 'rb'))
    a1,i1,c1=perform_logistic_regression(file,target)
    file=FileStorage(filename='f', stream=open('temp_anal/f', 'rb'))
    a2,i2,c2=perform_knn(file,target)
    file=FileStorage(filename='f', stream=open('temp_anal/f', 'rb'))
    a3,i3,_,c3=perform_dtree(file,target)
    file=FileStorage(filename='f', stream=open('temp_anal/f', 'rb'))
    a4,i4,c4=perform_naivebayes(file,target)
    file=FileStorage(filename='f', stream=open('temp_anal/f', 'rb'))
    a5,i5,c5=perform_svm(file,target)

    models_acc={}
    models = ['Logistic Regression', 'KNN', 'Decision Tree', 'Naive Bayes', 'SVM']
    accuracies = [a1, a2, a3, a4, a5]
    images=[i1,i2,i3,i4,i5]
    conf_matrices = [c1, c2, c3, c4, c5]
    for model, accuracy, conf_matrix,imgs in zip(models, accuracies, conf_matrices,images):
        rounded_accuracy = round(accuracy * 100, 2)
        correct = np.diag(conf_matrix).sum()
        total = np.sum(conf_matrix)
        wrong = total - correct

        models_acc[model] = [rounded_accuracy, imgs, [correct, wrong, total]]

    models_acc = dict(sorted(models_acc.items(), key=lambda item: item[1][0], reverse=True))
    # print(models_acc.keys())
    return(models_acc)



#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from io import BytesIO
import base64
import os
import pickle
from datetime import datetime

def save_model(model, metrics, coefficients, p_values, intercept, plots, 
               feature_encoders=None, target_encoder=None, original_target_values=None):
    """
    Save the model and its associated data to a file, including encoders
    """
    # Create unique filename
    filename = f'model_{int(datetime.now().timestamp())}.pkl'
    
    # Get the absolute path to the app directory
    app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Create saved_models directory in the app directory
    model_dir = os.path.join(app_dir, 'saved_models')
    os.makedirs(model_dir, exist_ok=True)
    
    # Full path for the model file
    filepath = os.path.join(model_dir, filename)
    
    # Prepare model data with original target values
    model_data = {
        'model': model,
        'metrics': metrics,
        'coefficients': coefficients,
        'p_values': p_values,
        'intercept': intercept,
        'plots': plots,
        'feature_encoders': feature_encoders,
        'target_encoder': target_encoder,
        'original_target_values': original_target_values  # Add this line
    }
    
    # Save the model
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        raise
    
    return filename

def get_plot_as_base64():
    image_stream = BytesIO()
    plt.savefig(image_stream, format='png', bbox_inches='tight')
    image_stream.seek(0)
    img_str = base64.b64encode(image_stream.read()).decode('utf-8')
    plt.close()
    return img_str

def perform_logistic_regression(file, target):
    # Read data
    df = pd.read_csv(file, index_col=False)
    
    # Initialize encoders dictionary
    feature_encoders = {}
    
    # Handle categorical features
    X = df.drop(columns=[target])
    if 'id' in X.columns:
        X = X.drop(columns=['id'])
    
    # Convert categorical columns to numeric
    for column in X.columns:
        if X[column].dtype == 'object' or X[column].dtype.name == 'category':
            encoder = LabelEncoder()
            X[column] = encoder.fit_transform(X[column].astype(str))
            feature_encoders[column] = encoder
    
     # Handle categorical target variable
    target_encoder = None
    y = df[target].values
    original_target_values = None
    if df[target].dtype == 'object' or df[target].dtype.name == 'category':
        target_encoder = LabelEncoder()
        # Store original unique values before encoding
        original_target_values = df[target].unique()
        y = target_encoder.fit_transform(y.astype(str))
    
    # Store feature names
    feature_names = X.columns
    
    # Convert to numpy array
    X = X.values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Fit model
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    conf_matrix = confusion_matrix(y_test, y_pred_test)
    classification_rep = classification_report(y_test, y_pred_test)
    
    # Initialize plots list
    plots = []
    
    # 1. Feature importance plot
    plt.figure(figsize=(10, 6))
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': abs(model.coef_[0])
    }).sort_values('importance', ascending=True)
    
    plt.barh(range(len(feature_importance)), feature_importance['importance'])
    plt.yticks(range(len(feature_importance)), feature_importance['feature'])
    plt.xlabel('Absolute Coefficient Value')
    plt.title('Feature Importance')
    plots.append(get_plot_as_base64())
    
    # 2. Correlation heatmap for numeric features only
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        plt.figure(figsize=(10, 8))
        correlation_matrix = numeric_df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Heatmap (Numeric Features Only)')
        plots.append(get_plot_as_base64())
    
    # 3. Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plots.append(get_plot_as_base64())
    
    # 4. ROC Curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plots.append(get_plot_as_base64())
    
    # Calculate p-values
    logit_model = sm.Logit(y_train, sm.add_constant(X_train_scaled))
    results = logit_model.fit(disp=0)
    p_values = results.pvalues[1:]
    
    # Create dictionaries
    coefficients = dict(zip(feature_names, model.coef_[0]))
    p_values_dict = dict(zip(feature_names, p_values))
    
    # Save model with encoders
    model_path = save_model(
        model=model,
        metrics={
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'roc_auc': roc_auc,
            'classification_report': classification_report
        },
        coefficients=coefficients,
        p_values=p_values_dict,
        intercept=model.intercept_[0],
        plots=plots,
        feature_encoders=feature_encoders,
        target_encoder=target_encoder,
        original_target_values=original_target_values  # Add this line
    )
    
    return (
        model_path,
        plots,
        {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'roc_auc': roc_auc,
            'classification_report': classification_report
        },
        coefficients,
        p_values_dict,
        model.intercept_[0]
    )

#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################

def perform_knn(file,target):
    df = pd.read_csv(file,index_col=False)
    y=df[target]
    target=[target]
    if 'id' in df.columns:
        target.append('id')
    X=df.drop(columns=target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)
    
    model = KNeighborsClassifier(n_neighbors=3)
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

def perform_dtree(file,target):
    df = pd.read_csv(file,index_col=False)
    y=df[target]
    target=[target]
    if 'id' in df.columns:
        target.append('id')
    X=df.drop(columns=target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)
    model=DecisionTreeClassifier()
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
    
    plt.figure(figsize=(10, 8))
    tree.plot_tree(model, feature_names=X_train.columns,filled=True)
    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    image_stream.seek(0)
    tree_str = base64.b64encode(image_stream.read()).decode('utf-8')
    return(accuracy,img_str,tree_str,conf_matrix)
#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################

def perform_naivebayes(file,target):
    df = pd.read_csv(file,index_col=False)
    y=df[target]
    target=[target]
    if 'id' in df.columns:
        target.append('id')
    X=df.drop(columns=target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)
    model=GaussianNB()
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

def perform_svm(file,target):
    df = pd.read_csv(file,index_col=False)
    y=df[target]
    target=[target]
    if 'id' in df.columns:
        target.append('id')
    X=df.drop(columns=target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)
    model=SVC(kernel='rbf')
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