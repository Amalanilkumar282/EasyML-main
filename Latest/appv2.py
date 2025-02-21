# app.py
import traceback
from flask import Flask, render_template, request, jsonify, redirect, url_for
import mysql.connector
import numpy as np
import pandas as pd
import ast
import pickle
from flask import send_file
import os
from models.classification import *
from models.regression import *
from werkzeug.datastructures import FileStorage

from flask import request, redirect, url_for, render_template
from werkzeug.datastructures import FileStorage
import pandas as pd
import os
from flask import session, flash
from werkzeug.security import check_password_hash, generate_password_hash
import logging

app = Flask(__name__)
app.config['MODEL_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models')


# Create a logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Set the log level for the logger

# Create a file handler to log messages to a file
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.INFO)

# Create a stream handler to log messages to the terminal
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

# Define a log message format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)



# Database Credentials
db_credentials = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'NoCodeML'
}

def authenticate_user(email, password):
    # Connect to the database
    db_connection = mysql.connector.connect(
        host=db_credentials['host'],
        user=db_credentials['user'],
        password=db_credentials['password'],
        database=db_credentials['database']
    )

    cursor = db_connection.cursor(dictionary=True)
    query = "SELECT * FROM users WHERE email = %s AND password = %s"
    cursor.execute(query, (email, password))
    user = cursor.fetchone()

    # Close the database connection
    db_connection.close()

    return user

@app.route('/login', methods=['POST'])
def login():
    email = request.form.get('email')
    password = request.form.get('password')

    user = authenticate_user(email, password)

    if user:
        return render_template('landing.html')  
    else:
        return render_template('sign-in.html', error='Invalid credentials')
    

@app.route('/sign-up')
def sign_up():
    return render_template('sign-up.html')

@app.route('/sign-in')
def sign_in():
    return render_template('sign-in.html')


@app.route('/models')
def model():
    return(render_template('models.html'))

@app.route('/reg')
def linear():
    file=FileStorage(filename='f', stream=open('tempsy/f', 'rb'))
    model, plot = perform_linear_regression(file)
    return render_template('linearreg.html', plot=plot, model=model)




@app.route('/multireg', methods=['GET', 'POST'])
def multilinear():
    if request.method == 'POST':
        file = FileStorage(filename='f', stream=open('tempsy/f', 'rb'))
        target = request.json.get('variable', '')
        model_path, plots, metrics, coefficients, p_values, intercept = perform_multiple_linear_regression(file, target)
        
        return render_template('multilinear_reg_sample.html',
                             plots=plots,
                             metrics=metrics,
                             coefficients=coefficients,
                             p_values=p_values,
                             intercept=intercept,
                             model_path=os.path.basename(model_path))
    else:
        # Handle GET request - redirect to feature selection
        return redirect(url_for('display_features', m='multireg'))

def save_model(model, metrics, coefficients, p_values, intercept, plots):
    """
    Save the model and its associated data to a file
    """
    import os
    import pickle
    from datetime import datetime
    
    # Create a unique filename based on timestamp
    filename = f'model_{int(datetime.now().timestamp())}.pkl'
    
    # Use the MODEL_FOLDER from app config
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'saved_models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
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
    
    return filename  # Return just the filename instead of full path

# In app.py - Update the download_model and predict_multiple routes

@app.route('/download_model/<filename>')
def download_model(filename):
    try:
        model_path = os.path.join(app.config['MODEL_FOLDER'], filename)
        if not os.path.exists(model_path):
            return render_template('error.html', 
                                error_message=f"Model file not found: {filename}. Please retrain the model.")
        return send_file(model_path,
                        mimetype='application/octet-stream',
                        as_attachment=True,
                        download_name=filename)
    except Exception as e:
        return render_template('error.html', 
                             error_message=f"Error downloading model: {str(e)}")

@app.route('/predict_multiple', methods=['POST'])
def predict_multiple():
    try:
        # Get the model filename and features from the form
        model_filename = request.form['model_path']
        features = request.form['features'].split(',')
        
        # Construct full model path
        model_path = os.path.join(app.config['MODEL_FOLDER'], model_filename)
        
        # Verify model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Load the saved model
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
            
        model = saved_data['model']
        
        # Collect input values
        values_dict = {}
        for feature in features:
            try:
                values_dict[feature] = float(request.form[feature])
            except ValueError:
                raise ValueError(f"Invalid value provided for feature '{feature}'")
        
        # Make prediction
        prediction = predict_new_values(model, features, values_dict)
        
        # Re-render the results page with the prediction
        return render_template(
            'multilinear_reg_sample.html',
            model_path=model_filename,  # Pass filename only
            metrics=saved_data['metrics'],
            coefficients=saved_data['coefficients'],
            p_values=saved_data['p_values'],
            intercept=saved_data['intercept'],
            plots=saved_data['plots'],
            prediction=prediction,
            input_values=values_dict
        )
        
    except Exception as e:
        return render_template('error.html', 
                             error_message=f"Error during prediction: {str(e)}")





@app.route('/predict_new', methods=['POST'])
def predict_new():
    file=FileStorage(filename='f', stream=open('tempsy/f', 'rb'))
    model, plot = perform_linear_regression(file)
    new_value = float(request.form['new_value'])
    new_value_reshaped = np.array([new_value]).reshape(-1, 1)
    prediction = model.predict(new_value_reshaped)
    return render_template('linearreg.html',plot=plot, prediction=prediction, model=model, new_value=new_value)

@app.route('/logreg', methods=['GET', 'POST'])
def logistic():
    if request.method == 'POST':
        file = FileStorage(filename='f', stream=open('tempsy/f', 'rb'))
        target = request.json.get('variable', '')
        
        try:
            # Get model path and other results
            model_path, plots, metrics, coefficients, p_values, intercept = perform_logistic_regression(file, target)
            
            # Wait until the model file is actually saved
            model_full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models', model_path)
            
            # Ensure the saved_models directory exists
            os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models'), exist_ok=True)
            
            # Check if file exists
            if not os.path.exists(model_full_path):
                raise FileNotFoundError(f"Model file not found at {model_full_path}")
            
            # Get feature names from the saved model
            with open(model_full_path, 'rb') as f:
                saved_data = pickle.load(f)
                feature_names = list(saved_data['coefficients'].keys())
            
            # Format metrics for display
            formatted_metrics = {
                'accuracy': f"{metrics['test_accuracy']*100:.2f}%",
                'train_accuracy': f"{metrics['train_accuracy']*100:.2f}%",
                'roc_auc': f"{metrics['roc_auc']:.2f}"
            }
            
            return render_template('logistic.html',
                                 plots=plots,
                                 metrics=formatted_metrics,
                                 coefficients=coefficients,
                                 p_values=p_values,
                                 intercept=intercept,
                                 model_path=model_path,
                                 feature_names=feature_names)
                                 
        except Exception as e:
            # Log the error (you should set up proper logging)
            print(f"Error in logistic regression: {str(e)}")
            return f"An error occurred: {str(e)}", 500
            
    else:
        # Handle GET request
        return redirect(url_for('display_features', m='logreg'))
    

    #Logistic regression new value prediction
@app.route('/predict_logistic', methods=['POST'])
def predict_logistic():
    try:
        # Get the model filename and features from the form
        model_filename = request.form['model_path']
        features = request.form['features'].split(',')
        
        # Construct full model path
        model_path = os.path.join(app.config['MODEL_FOLDER'], model_filename)
        
        # Verify model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Load the saved model
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
            
        model = saved_data['model']
        original_target_values = saved_data['original_target_values']
        feature_encoders = saved_data.get('feature_encoders', {})
        target_encoder = saved_data.get('target_encoder', None)
        scaler = saved_data.get('scaler', None)  # Assuming you saved the scaler

        # Collect input values with proper encoding and validation
        values_dict = {}
        raw_values = {}  # Store original values for display
        for feature in features:
            input_value = request.form[feature].strip()
            raw_values[feature] = input_value  # Store original value
            
            # Handle categorical features
            if feature in feature_encoders:
                encoder = feature_encoders[feature]
                try:
                    # Try to convert to numerical value first (in case encoded value was entered)
                    numerical_value = float(input_value)
                    if numerical_value in encoder.classes_:
                        values_dict[feature] = numerical_value
                    else:
                        # If not a valid numerical value, try string transformation
                        encoded_value = encoder.transform([input_value])[0]
                        values_dict[feature] = encoded_value
                except ValueError:
                    # Handle categorical string values
                    try:
                        encoded_value = encoder.transform([input_value])[0]
                        values_dict[feature] = encoded_value
                    except ValueError:
                        valid_values = list(encoder.classes_)
                        raise ValueError(
                            f"Invalid value '{input_value}' for feature '{feature}'. "
                            f"Valid options are: {', '.join(map(str, valid_values))}"
                        )
            else:
                # Handle numerical features
                try:
                    values_dict[feature] = float(input_value)
                except ValueError:
                    raise ValueError(f"Invalid numerical value '{input_value}' for feature '{feature}'")

        # Scale the input data if scaler exists
        if scaler:
            input_array = np.array([[values_dict[feature] for feature in features]])
            input_array = scaler.transform(input_array)
        else:
            input_array = np.array([[values_dict[feature] for feature in features]])

        # Make prediction
        prediction_proba = model.predict_proba(input_array)[0]
        prediction = model.predict(input_array)[0]

        # Map the numeric prediction to actual class name from original values
        if target_encoder:
            actual_class_name = target_encoder.inverse_transform([prediction])[0]
        else:
            actual_class_name = str(prediction)

        # Create probabilities dictionary with actual class names
        if target_encoder and original_target_values is not None:
            probabilities = {
                str(cls): f"{prob * 100:.2f}%"
                for cls, prob in zip(original_target_values, prediction_proba)
            }
        else:
            probabilities = {
                str(i): f"{prob * 100:.2f}%"
                for i, prob in enumerate(prediction_proba)
            }

        # Format metrics properly
        formatted_metrics = {
            'accuracy': f"{saved_data['metrics']['test_accuracy']*100:.2f}%",
            'train_accuracy': f"{saved_data['metrics']['train_accuracy']*100:.2f}%",
            'roc_auc': f"{saved_data['metrics']['roc_auc']:.2f}"
        }

        # Re-render the results page with the prediction
        return render_template(
            'logistic.html',
            model_path=model_filename,
            metrics=formatted_metrics,
            coefficients=saved_data['coefficients'],
            p_values=saved_data['p_values'],
            plots=saved_data.get('plots', []),
            prediction=actual_class_name,
            final_class_name=actual_class_name,
            probability=max(prediction_proba) * 100,
            probabilities=probabilities,
            input_values=raw_values,
            class_mapping=original_target_values,
            feature_names=features  # Pass feature names for table
        )
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        traceback.print_exc()
        return render_template('error.html', 
                             error_message=f"Prediction error: {str(e)}")

def predict_new_values_logistic(model, features, values_dict):
    """Helper function to make predictions with logistic regression model"""
    try:
        # Create input array in the correct order
        input_array = np.array([[values_dict[feature] for feature in features]])
        
        # Make prediction
        prediction_proba = model.predict_proba(input_array)[0]
        prediction = model.predict(input_array)[0]
        
        return prediction, prediction_proba
    except Exception as e:
        print(f"Debug - Prediction error: {str(e)}")
        raise





@app.route('/knn', methods=['GET', 'POST'])
def knn_f():
    if request.method == 'POST':
        file = FileStorage(filename='f', stream=open('tempsy/f', 'rb'))
        target = request.json.get('variable', '')
        try:
            model_path, plots, metrics, _, _, _ = perform_knn(file, target)
            
            model_full_path = os.path.join(app.config['MODEL_FOLDER'], model_path)
            if not os.path.exists(model_full_path):
                raise FileNotFoundError(f"Model file not found at {model_full_path}")
            
            with open(model_full_path, 'rb') as f:
                saved_data = pickle.load(f)
                feature_names = saved_data.get('feature_names', [])
            
            formatted_metrics = {
                'accuracy': metrics['accuracy'],
                'roc_auc': metrics.get('roc_auc')
            }
            
            return render_template('knn.html',
                                 plots=plots,
                                 metrics=formatted_metrics,
                                 model_path=model_path,
                                 feature_names=feature_names)
        except Exception as e:
            app.logger.error(f"Error in KNN: {str(e)}")
            return render_template('error.html', error_message=str(e))
    else:
        return redirect(url_for('display_features', m='knn'))

@app.route('/predict_knn', methods=['POST'])
def predict_knn():
    try:
        # Load model components
        model_filename = request.form['model_path']
        model_path = os.path.join(app.config['MODEL_FOLDER'], model_filename)
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        scaler = model_data['scaler']
        feature_encoders = model_data.get('feature_encoders', {})
        target_encoder = model_data.get('target_encoder')
        feature_names = model_data['feature_names']
        
        # Process input features
        encoded_features = []
        input_values = {}
        
        for feature in feature_names:
            raw_value = request.form[feature].strip()
            input_values[feature] = raw_value  # Store original value

            if feature in feature_encoders:
                # Handle categorical feature
                encoder = feature_encoders[feature]
                try:
                    encoded_value = encoder.transform([raw_value])[0]
                except ValueError:
                    valid_classes = list(encoder.classes_)
                    raise ValueError(
                        f"Invalid value '{raw_value}' for {feature}. "
                        f"Valid options: {', '.join(valid_classes)}"
                    )
                encoded_features.append(encoded_value)
            else:
                # Handle numerical feature
                try:
                    encoded_features.append(float(raw_value))
                except ValueError:
                    raise ValueError(f"Invalid number for {feature}: {raw_value}")

        # Scale and predict
        scaled_input = scaler.transform([encoded_features])
        prediction = model.predict(scaled_input)[0]
        probabilities = model.predict_proba(scaled_input)[0] if hasattr(model, 'predict_proba') else None
        
        # Decode prediction
        if target_encoder:
            class_name = target_encoder.inverse_transform([prediction])[0]
            if probabilities is not None:
                prob_dict = {
                    str(cls): f"{prob*100:.1f}%"
                    for cls, prob in zip(model_data['original_target_values'], probabilities)
                }
            else:
                prob_dict = None
        else:
            class_name = str(prediction)
            prob_dict = None

        return render_template('knn.html',
                             model_path=model_filename,
                             prediction=class_name,
                             probabilities=prob_dict,
                             input_values=input_values,
                             feature_names=feature_names,
                             metrics=model_data['metrics'],
                             plots=model_data['plots'])

    except Exception as e:
        return render_template('error.html', 
                             error_message=f"Prediction Error: {str(e)}")
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################

@app.route('/dtree', methods=['GET', 'POST'])
def decision_tree():
    if request.method == 'POST':
        file = FileStorage(filename='f', stream=open('tempsy/f', 'rb'))
        target = request.json.get('variable', '')
        try:
            model_path, plots, metrics, _, _, _ = perform_dtree(file, target)
            
            model_full_path = os.path.join(app.config['MODEL_FOLDER'], model_path)
            if not os.path.exists(model_full_path):
                raise FileNotFoundError(f"Model file not found at {model_full_path}")
            
            with open(model_full_path, 'rb') as f:
                saved_data = pickle.load(f)
                feature_names = saved_data.get('feature_names', [])
            
            # Extract confusion matrix metrics
            conf_matrix = np.array(metrics['classification_report']['confusion_matrix'])
            correct = conf_matrix.diagonal().sum()
            total = conf_matrix.sum()
            wrong = total - correct
            conf = [int(correct), int(wrong), int(total)]
            
            # Format metrics
            formatted_metrics = {
                'accuracy': metrics['accuracy'],
                'precision': metrics['classification_report']['weighted avg']['precision'],
                'recall': metrics['classification_report']['weighted avg']['recall']
            }
            
            return render_template('dtree.html',
                                plots=plots,
                                metrics=formatted_metrics,
                                model_path=model_path,
                                feature_names=feature_names,
                                conf=conf,
                                precision=formatted_metrics['precision'] * 100,
                                recall=formatted_metrics['recall'] * 100,
                                acc=formatted_metrics['accuracy'] * 100)
        except Exception as e:
            app.logger.error(f"Error in Decision Tree: {str(e)}")
            return render_template('error.html', error_message=str(e))
    else:
        return redirect(url_for('display_features', m='dtree'))

@app.route('/predict_dtree', methods=['POST'])
def predict_dtree():
    try:
        # Load model components
        model_filename = request.form['model_path']
        model_path = os.path.join(app.config['MODEL_FOLDER'], model_filename)
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        feature_encoders = model_data.get('feature_encoders', {})
        target_encoder = model_data.get('target_encoder')
        feature_names = model_data['feature_names']
        metrics = model_data['metrics']
        plots = model_data.get('plots', [])
        
        # Process input features
        encoded_features = []
        input_values = {}
        
        for feature in feature_names:
            raw_value = request.form[feature].strip()
            input_values[feature] = raw_value

            if feature in feature_encoders:
                encoder = feature_encoders[feature]
                try:
                    encoded_value = encoder.transform([raw_value])[0]
                except ValueError:
                    valid_classes = list(encoder.classes_)
                    raise ValueError(
                        f"Invalid value '{raw_value}' for {feature}. "
                        f"Valid options: {', '.join(valid_classes)}"
                    )
                encoded_features.append(encoded_value)
            else:
                try:
                    encoded_features.append(float(raw_value))
                except ValueError:
                    raise ValueError(f"Invalid number for {feature}: {raw_value}")

        # Make prediction
        prediction = model.predict([encoded_features])[0]
        probabilities = model.predict_proba([encoded_features])[0]
        
        # Decode prediction
        if target_encoder:
            class_name = target_encoder.inverse_transform([prediction])[0]
            prob_dict = {
                str(cls): f"{prob*100:.1f}%"
                for cls, prob in zip(model_data['original_target_values'], probabilities)
            }
        else:
            class_name = str(prediction)
            prob_dict = None

        # Get confusion matrix metrics
        conf_matrix = np.array(metrics['classification_report']['confusion_matrix'])
        correct = conf_matrix.diagonal().sum()
        total = conf_matrix.sum()
        wrong = total - correct
        conf = [int(correct), int(wrong), int(total)]

        # Format metrics for display
        formatted_metrics = {
            'accuracy': metrics['accuracy'],
            'precision': metrics['classification_report']['weighted avg']['precision'],
            'recall': metrics['classification_report']['weighted avg']['recall']
        }

        return render_template('dtree.html',
                             model_path=model_filename,
                             prediction=class_name,
                             probabilities=prob_dict,
                             input_values=input_values,
                             feature_names=feature_names,
                             metrics=formatted_metrics,
                             plots=plots,
                             tree_plot=plots[1],
                             conf_matrix_plot=plots[0],
                             feature_importance_plot=plots[2],
                             conf=conf,
                             precision=formatted_metrics['precision'] * 100,
                             recall=formatted_metrics['recall'] * 100,
                             acc=formatted_metrics['accuracy'] * 100)

    except Exception as e:
        return render_template('error.html', 
                             error_message=f"Prediction Error: {str(e)}")

@app.route('/naivebayes', methods=['GET', 'POST'])
def naive_bayes():
    if request.method == 'POST':
        file = FileStorage(filename='f', stream=open('tempsy/f', 'rb'))
        target = request.json.get('variable', '')
        try:
            model_path, plots, metrics, _, _, _ = perform_naivebayes(file, target)
            
            model_full_path = os.path.join(app.config['MODEL_FOLDER'], model_path)
            if not os.path.exists(model_full_path):
                raise FileNotFoundError(f"Model file not found at {model_full_path}")
            
            with open(model_full_path, 'rb') as f:
                saved_data = pickle.load(f)
                feature_names = saved_data.get('feature_names', [])
            
            # Extract confusion matrix directly from metrics
            conf_matrix = np.array(metrics['classification_report']['confusion_matrix'])
            
            # Calculate confusion matrix metrics
            correct = conf_matrix.diagonal().sum()
            total = conf_matrix.sum()
            wrong = total - correct
            conf = [int(correct), int(wrong), int(total)]
            
            # Format metrics
            formatted_metrics = {
                'accuracy': metrics['accuracy'],
                'precision': metrics['classification_report']['weighted avg']['precision'],
                'recall': metrics['classification_report']['weighted avg']['recall']
            }
            
            return render_template('naivebayes.html',
                                plots=plots,
                                metrics=formatted_metrics,
                                model_path=model_path,
                                feature_names=feature_names,
                                conf=conf,
                                precision=formatted_metrics['precision'] * 100,
                                recall=formatted_metrics['recall'] * 100,
                                acc=formatted_metrics['accuracy'] * 100)
        except Exception as e:
            app.logger.error(f"Error in Naive Bayes: {str(e)}")
            return render_template('error.html', error_message=str(e))
    else:
        return redirect(url_for('display_features', m='naivebayes'))

@app.route('/predict_naivebayes', methods=['POST'])
def predict_naivebayes():
    try:
        # Load model components
        model_filename = request.form['model_path']
        model_path = os.path.join(app.config['MODEL_FOLDER'], model_filename)
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        scaler = model_data['scaler']
        feature_encoders = model_data.get('feature_encoders', {})
        target_encoder = model_data.get('target_encoder')
        feature_names = model_data['feature_names']
        metrics = model_data['metrics']
        plots = model_data.get('plots', [])
        
        # Process input features
        encoded_features = []
        input_values = {}
        
        for feature in feature_names:
            raw_value = request.form[feature].strip()
            input_values[feature] = raw_value

            if feature in feature_encoders:
                encoder = feature_encoders[feature]
                try:
                    encoded_value = encoder.transform([raw_value])[0]
                except ValueError:
                    valid_classes = list(encoder.classes_)
                    raise ValueError(
                        f"Invalid value '{raw_value}' for {feature}. "
                        f"Valid options: {', '.join(valid_classes)}"
                    )
                encoded_features.append(encoded_value)
            else:
                try:
                    encoded_features.append(float(raw_value))
                except ValueError:
                    raise ValueError(f"Invalid number for {feature}: {raw_value}")

        # Scale and predict
        scaled_input = scaler.transform([encoded_features])
        prediction = model.predict(scaled_input)[0]
        probabilities = model.predict_proba(scaled_input)[0]
        
        # Decode prediction
        if target_encoder:
            class_name = target_encoder.inverse_transform([prediction])[0]
            prob_dict = {
                str(cls): f"{prob*100:.1f}%"
                for cls, prob in zip(model_data['original_target_values'], probabilities)
            }
        else:
            class_name = str(prediction)
            prob_dict = None

        # Get confusion matrix metrics
        conf_matrix = np.array(metrics['classification_report']['confusion_matrix'])
        correct = conf_matrix.diagonal().sum()
        total = conf_matrix.sum()
        wrong = total - correct
        conf = [int(correct), int(wrong), int(total)]

        # Format metrics for display
        formatted_metrics = {
            'accuracy': metrics['accuracy'],
            'precision': metrics['classification_report']['weighted avg']['precision'],
            'recall': metrics['classification_report']['weighted avg']['recall']
        }

        return render_template('naivebayes.html',
                             model_path=model_filename,
                             prediction=class_name,
                             probabilities=prob_dict,
                             input_values=input_values,
                             feature_names=feature_names,
                             metrics=formatted_metrics,
                             plots=plots,
                             conf=conf,
                             precision=formatted_metrics['precision'] * 100,
                             recall=formatted_metrics['recall'] * 100,
                             acc=formatted_metrics['accuracy'] * 100)

    except Exception as e:
        return render_template('error.html', 
                             error_message=f"Prediction Error: {str(e)}")

@app.route('/svm', methods=['GET', 'POST'])
def svm():
    if request.method == 'POST':
        file = FileStorage(filename='f', stream=open('tempsy/f', 'rb'))
        target = request.json.get('variable', '')
        try:
            model_path, plots, metrics, _, _, _ = perform_svm(file, target)
            
            model_full_path = os.path.join(app.config['MODEL_FOLDER'], model_path)
            if not os.path.exists(model_full_path):
                raise FileNotFoundError(f"Model file not found at {model_full_path}")
            
            with open(model_full_path, 'rb') as f:
                saved_data = pickle.load(f)
                feature_names = saved_data.get('feature_names', [])
            
            # Extract confusion matrix directly from metrics
            conf_matrix = np.array(metrics['classification_report']['confusion_matrix'])
            
            # Calculate confusion matrix metrics
            correct = conf_matrix.diagonal().sum()
            total = conf_matrix.sum()
            wrong = total - correct
            conf = [int(correct), int(wrong), int(total)]
            
            # Format metrics
            formatted_metrics = {
                'accuracy': metrics['accuracy'],
                'precision': metrics['classification_report']['weighted avg']['precision'],
                'recall': metrics['classification_report']['weighted avg']['recall']
            }
            
            return render_template('svm.html',
                                 plots=plots,
                                 metrics=formatted_metrics,
                                 model_path=model_path,
                                 feature_names=feature_names,
                                 conf=conf,
                                 precision=formatted_metrics['precision'] * 100,
                                 recall=formatted_metrics['recall'] * 100,
                                 acc=formatted_metrics['accuracy'] * 100)
        except Exception as e:
            app.logger.error(f"Error in SVM: {str(e)}")
            return render_template('error.html', error_message=str(e))
    else:
        return redirect(url_for('display_features', m='svm'))

@app.route('/predict_svm', methods=['POST'])
def predict_svm():
    try:
        # Load model components
        model_filename = request.form['model_path']
        model_path = os.path.join(app.config['MODEL_FOLDER'], model_filename)
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        scaler = model_data['scaler']
        feature_encoders = model_data.get('feature_encoders', {})
        target_encoder = model_data.get('target_encoder')
        feature_names = model_data['feature_names']
        metrics = model_data['metrics']
        plots = model_data.get('plots', [])  # Get plots from saved model data
        
        # Process input features
        encoded_features = []
        input_values = {}
        
        for feature in feature_names:
            raw_value = request.form[feature].strip()
            input_values[feature] = raw_value

            if feature in feature_encoders:
                encoder = feature_encoders[feature]
                try:
                    encoded_value = encoder.transform([raw_value])[0]
                except ValueError:
                    valid_classes = list(encoder.classes_)
                    raise ValueError(
                        f"Invalid value '{raw_value}' for {feature}. "
                        f"Valid options: {', '.join(valid_classes)}"
                    )
                encoded_features.append(encoded_value)
            else:
                try:
                    encoded_features.append(float(raw_value))
                except ValueError:
                    raise ValueError(f"Invalid number for {feature}: {raw_value}")

        # Scale and predict
        scaled_input = scaler.transform([encoded_features])
        prediction = model.predict(scaled_input)[0]
        probabilities = model.predict_proba(scaled_input)[0] if hasattr(model, 'predict_proba') else None
        
        # Decode prediction
        if target_encoder:
            class_name = target_encoder.inverse_transform([prediction])[0]
            if probabilities is not None:
                prob_dict = {
                    str(cls): f"{prob*100:.1f}%"
                    for cls, prob in zip(model_data['original_target_values'], probabilities)
                }
            else:
                prob_dict = None
        else:
            class_name = str(prediction)
            prob_dict = None

        # Get confusion matrix from metrics
        conf_matrix = np.array(metrics['classification_report']['confusion_matrix'])
        correct = conf_matrix.diagonal().sum()
        total = conf_matrix.sum()
        wrong = total - correct
        conf = [int(correct), int(wrong), int(total)]

        # Format metrics for display
        formatted_metrics = {
            'accuracy': metrics['accuracy'],
            'precision': metrics['classification_report']['weighted avg']['precision'],
            'recall': metrics['classification_report']['weighted avg']['recall']
        }

        return render_template('svm.html',
                             model_path=model_filename,
                             prediction=class_name,
                             probabilities=prob_dict,
                             input_values=input_values,
                             feature_names=feature_names,
                             metrics=formatted_metrics,
                             plots=plots,  # Pass the plots
                             feature_plot=plots[1] if len(plots) > 1 else None,  # Feature distribution plot
                             plot=plots[0] if plots else None,  # Confusion matrix plot
                             conf=conf,
                             precision=formatted_metrics['precision'] * 100,
                             recall=formatted_metrics['recall'] * 100,
                             acc=formatted_metrics['accuracy'] * 100)

    except Exception as e:
        return render_template('error.html', 
                             error_message=f"Prediction Error: {str(e)}")





# @app.route('/analysis',methods=['POST'])
# def analysis():

#     file=FileStorage(filename='f', stream=open('tempsy/f', 'rb'))
#     target = request.json.get('variable', '')
#     models=perform_analysis(file,target)
#     best_model = list(models.keys())[0]
#     best_model_info = models[best_model]
#     models.pop(best_model)
#     return render_template('analysis-classification.html',models=models,best_model=best_model,best_model_info=best_model_info)
# Flask routes
# from model_analysis import analyze_dataset
# Flask routes
from models.model_analyzer import *
@app.route('/analysis', methods=['POST'])
def analyze():
    try:
        file_path = os.path.join('tempsy', 'f')
        result = analyze_dataset(file_path)
        
        if result["status"] == "error":
            return jsonify(result), 400
        
        return render_template('analysis_result.html', 
                             model=result["best_model"], 
                             target=result["target_column"],
                             details=result["model_details"])
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500







@app.route('/signup', methods=['POST'])
def signup():
    try:
        # Get form data
        full_name = request.form['full_name']
        phone_no = request.form['phone_no']
        dob = request.form['dob']
        email = request.form['email']
        password = request.form['password']

        # Connect to the database
        db_connection = mysql.connector.connect(
            host=db_credentials['host'],
            user=db_credentials['user'],
            password=db_credentials['password'],
            database=db_credentials['database']
        )

        cursor = db_connection.cursor()

        # Insert data into the database
        
        query = "INSERT INTO users (fullname, phone_no, dob, email, password) VALUES (%s, %s, %s, %s, %s)"
        cursor.execute(query, (full_name, phone_no, dob, email, password))

        # Commit changes and close the database connection
        db_connection.commit()
        db_connection.close()

        # Return success message
        return render_template('sign-in.html')

    except Exception as e:
        # Return error message
        return jsonify({'success': False, 'message': str(e)})
    
# @app.route('/features')
# def display_features():
#     try:
#         # Read the CSV file
#         file_path = 'tempsy/f'
#         df = pd.read_csv(file_path)

#         # Get column names, data types, and number of distinct values
#         columns_info = []
#         for column in df.columns:
#             i=1
#             column_info = {
#                 'index':i,
#                 'name': column,
#                 'datatype': str(df[column].dtype),
#                 'distinct_values': df[column].nunique()
#             }
#             i+=1
#             columns_info.append(column_info)

#         return render_template('features.html', columns_info=columns_info)

#     except Exception as e:
#         # Handle any exceptions
#         return render_template('error.html', error=str(e))

@app.route('/features')
def display_features():
    # Your existing display_features route code
    file = FileStorage(filename='f', stream=open('tempsy/f', 'rb'))
    df = pd.read_csv(file)
    columns_info = []
    for column in df.columns:
        info = {
            'name': column,
            'datatype': str(df[column].dtype),
            'distinct_values': df[column].nunique()
        }
        columns_info.append(info)
    
    return render_template('features.html', columns_info=columns_info)

    
# Main route
@app.route('/')
def index():
    # return render_template('sign-in.html')
    return render_template('home.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    temp_folder = 'tempsy'
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    file_path = os.path.join(temp_folder, 'f')
    file.save(file_path)
    # file=pd.read_csv(file)
    # file.to_csv('temp/file.csv')
    return render_template('models.html')


app.run(debug=True, use_reloader=True, port=5004)