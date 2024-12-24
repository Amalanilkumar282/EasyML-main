# app.py
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

app = Flask(__name__)
app.config['MODEL_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models')


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




@app.route('/multireg')
def multilinear():
    file = FileStorage(filename='f', stream=open('tempsy/f', 'rb'))
    model_path, plots, metrics, coefficients, p_values, intercept = perform_multiple_linear_regression(file)
    
    return render_template('multilinear_reg_sample.html',
                         plots=plots,
                         metrics=metrics,
                         coefficients=coefficients,
                         p_values=p_values,
                         intercept=intercept,
                         model_path=os.path.basename(model_path))

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

@app.route('/logreg',methods=['POST'])
def logistic():
    file=FileStorage(filename='f', stream=open('tempsy/f', 'rb'))
    target = request.json.get('variable', '')
    acc,plot,confm = perform_logistic_regression(file,target)
    acc=round(acc*100,2)
    correct=confm.diagonal().sum()
    total=confm.sum()
    wrong=total-correct
    conf=[correct,wrong,total]
    precision=round((confm[0][0]/(confm[0][0]+confm[0][1]))*100,2)
    recall=round((confm[0][0]/(confm[0][0]+confm[1][0]))*100,2)
    return render_template('logistic.html',acc=acc, plot=plot, conf=conf,precision=precision,recall=recall)


@app.route('/knn',methods=['POST'])
def knn_f():

    file=FileStorage(filename='f', stream=open('tempsy/f', 'rb'))
    target = request.json.get('variable', '')
    acc,plot,confm = perform_knn(file,target)
    acc=round(acc*100,2)
    correct=confm.diagonal().sum()
    total=confm.sum()
    wrong=total-correct
    conf=[correct,wrong,total]
    precision=round((confm[0][0]/(confm[0][0]+confm[0][1]))*100,2)
    recall=round((confm[0][0]/(confm[0][0]+confm[1][0]))*100,2)
    return render_template('knn.html',acc=acc, plot=plot, conf=conf,precision=precision,recall=recall)

@app.route('/dtree',methods=['POST'])
def decision_tree():

    file=FileStorage(filename='f', stream=open('tempsy/f', 'rb'))
    target = request.json.get('variable', '')
    acc,plot,tree,confm = perform_dtree(file,target)
    acc=round(acc*100,2)
    correct=confm.diagonal().sum()
    total=confm.sum()
    wrong=total-correct
    conf=[correct,wrong,total]
    precision=round((confm[0][0]/(confm[0][0]+confm[0][1]))*100,2)
    recall=round((confm[0][0]/(confm[0][0]+confm[1][0]))*100,2)
    return render_template('dtree.html',acc=acc,tree=tree, plot=plot, conf=conf,precision=precision,recall=recall)

@app.route('/naivebayes',methods=['POST'])
def naive_bayes():

    file=FileStorage(filename='f', stream=open('tempsy/f', 'rb'))
    target = request.json.get('variable', '')
    acc,plot,confm = perform_naivebayes(file,target)
    acc=round(acc*100,2)
    correct=confm.diagonal().sum()
    total=confm.sum()
    wrong=total-correct
    conf=[correct,wrong,total]
    precision=round((confm[0][0]/(confm[0][0]+confm[0][1]))*100,2)
    recall=round((confm[0][0]/(confm[0][0]+confm[1][0]))*100,2)
    return render_template('naivebayes.html',acc=acc, plot=plot, conf=conf,precision=precision,recall=recall)

@app.route('/svm',methods=['POST'])
def svm():

    file=FileStorage(filename='f', stream=open('tempsy/f', 'rb'))
    target = request.json.get('variable', '')
    acc,plot,confm = perform_svm(file,target)
    acc=round(acc*100,2)
    correct=confm.diagonal().sum()
    total=confm.sum()
    wrong=total-correct
    conf=[correct,wrong,total]
    precision=round((confm[0][0]/(confm[0][0]+confm[0][1]))*100,2)
    recall=round((confm[0][0]/(confm[0][0]+confm[1][0]))*100,2)
    return render_template('svm.html',acc=acc, plot=plot, conf=conf,precision=precision,recall=recall)


@app.route('/analysis',methods=['POST'])
def analysis():

    file=FileStorage(filename='f', stream=open('tempsy/f', 'rb'))
    target = request.json.get('variable', '')
    models=perform_analysis(file,target)
    best_model = list(models.keys())[0]
    best_model_info = models[best_model]
    models.pop(best_model)
    return render_template('analysis-classification.html',models=models,best_model=best_model,best_model_info=best_model_info)

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
    
@app.route('/features')
def display_features():
    try:
        # Read the CSV file
        file_path = 'tempsy/f'
        df = pd.read_csv(file_path)

        # Get column names, data types, and number of distinct values
        columns_info = []
        for column in df.columns:
            i=1
            column_info = {
                'index':i,
                'name': column,
                'datatype': str(df[column].dtype),
                'distinct_values': df[column].nunique()
            }
            i+=1
            columns_info.append(column_info)

        return render_template('features.html', columns_info=columns_info)

    except Exception as e:
        # Handle any exceptions
        return render_template('error.html', error=str(e))
    
# Main route
@app.route('/')
def index():
    return render_template('sign-in.html')

# Upload route
# @app.route('/linearreg', methods=['POST'])
# def linear_reg():
#     if file is None:
#         return render_template('models.html', error='No file uploaded')

#     model, plot = perform_linear_regression(file)
#     return render_template('linearreg.html', plot=plot, model=model)

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


# @app.route('/predict', methods=['POST'])
# def predict():
#     new_value = float(request.form['new_value'])  # Get the new value as a float
#     # Create a new instance of LinearRegression
#     model = LinearRegression()

#     # Set the parameters for the new instance
#     model.intercept_ = request.form['inter']
#     model.intercept_=float(model.intercept_.strip('[]'))
#     model.coef_ = request.form['coef']
#     outer_list = ast.literal_eval(model.coef_)
#     inner_list_floats = [float(item) for item in outer_list[0]]
#     model.coef_ = np.array([inner_list_floats])
#     # Reshape the new value to fit the model's input requirements
#     new_value_reshaped = np.array([new_value]).reshape(-1, 1)

#     # Use the trained model to predict the new value
#     prediction = model.predict(new_value_reshaped)


#     return render_template('linearreg.html', prediction=prediction, model=model, new_value=new_value)

app.run(debug=True, use_reloader=True, port=5004)