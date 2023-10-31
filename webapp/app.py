# Standard library imports
import os
import pickle

# Third-party imports
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from pgmpy.inference import VariableElimination

app = Flask(__name__)
app.secret_key = 'a3c2b1a0c9f6e8d7b5e4f3a2c1e0d9f8'

# Define the path to the saved model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'saved_model.pkl')

# Load your trained model
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


def categorize_glucose_level(value):
    if value < 90:
        return 0
    elif 90 <= value < 130:
        return 1
    else:
        return 2


# Function to categorize bmi
def categorize_bmi(value):
    if value < 18.5:
        return 0
    elif 18.5 <= value < 24.9:
        return 1
    elif 24.9 <= value < 29.9:
        return 2
    else:
        return 3


# Function to categorize age
def categorize_age(value):
    return int(value // 10)


def preprocess_input_data(data_list):
    # Extract values based on known order
    gender = data_list[0] or "Unknown"
    age = data_list[1] or None
    hypertension = data_list[2] or 0
    heart_disease = data_list[3] or 0
    ever_married = data_list[4] or "No"
    work_type = data_list[5] or "Unknown"
    Residence_type = data_list[6] or "Unknown"
    avg_glucose_level = data_list[7] or None
    bmi = data_list[8] or None
    smoking_status = data_list[9] or "Unknown"

    # Gender validation and mapping
    gender = {'Male': 0, 'Female': 1, 'Unknown': None}.get(gender, None)

    # ever_married validation and mapping
    ever_married = {'Yes': 1, 'No': 0, 'Unknown': None}.get(ever_married, None)

    # work_type validation
    work_types = {'Private': 'isPrivate', 'Self-employed': 'isSelfEmployed', 'Govt_job': 'isGovt'}
    isPrivate, isSelfEmployed, isGovt = [1 if work_type == wt else 0 for wt in work_types.keys()]

    # Residence_type validation and mapping
    Residence_type = {'Urban': 1, 'Rural': 0, 'Unknown': None}.get(Residence_type, None)

    # Smoking_status validation and mapping
    smoking_status = {'never smoked': 0, 'formerly smoked': 1, 'smokes': 2, 'Unknown': None}.get(smoking_status, None)

    # Categorize avg_glucose_level, bmi, and age if they are not None
    avg_glucose_level = categorize_glucose_level(avg_glucose_level) if avg_glucose_level is not None else None
    bmi = categorize_bmi(bmi) if bmi is not None else None
    age = categorize_age(age) if age is not None else None

    # Return the processed data as a list
    return [gender, age, hypertension, heart_disease, ever_married, isPrivate, isSelfEmployed, Residence_type,
            avg_glucose_level, bmi, smoking_status]


def extract_data_from_form():
    return [
        request.form.get('gender'),
        float(request.form.get('age')),
        int(request.form.get('hypertension')),
        int(request.form.get('heart_disease')),
        request.form.get('ever_married'),
        request.form.get('work_type'),
        request.form.get('Residence_type'),
        float(request.form.get('avg_glucose_level')),
        float(request.form.get('bmi')),
        request.form.get('smoking_status')
    ]


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Extract data from form
            data_list = extract_data_from_form()

            # Preprocess data
            processed_data = preprocess_input_data(data_list)

            # Get prediction
            prediction = get_prediction(processed_data)
            return render_template('index.html', prediction=prediction)

        except ValueError as e:
            flash(str(e))
            return redirect(url_for('index'))

    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.is_json:
            # Get data from JSON POST request (API call)
            data_list = request.get_json(force=True)
        else:
            # Get data from form POST request (Webpage)
            data_list = extract_data_from_form()

        # Preprocess data
        processed_data = preprocess_input_data(data_list)

        # Get prediction
        prediction = get_prediction(processed_data)

        if request.is_json:
            # Return result for API
            return jsonify({"stroke_probability": prediction})
        else:
            # Return result for webpage
            return render_template('index.html', prediction=prediction)

    except ValueError as e:
        if request.is_json:
            # Return error message for API
            return jsonify({"error": str(e)}), 400  # 400 Bad Request
        else:
            # Return error message for webpage
            flash(str(e))
            return redirect(url_for('index'))


def get_prediction(processed_data):
    # Map the processed data to their respective keys
    keys = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'isPrivate', 'isSelfEmployed',
            'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']

    # Create a dictionary from the processed data for inference, only include non-None values
    data_dict = {key: value for key, value in zip(keys, processed_data) if value is not None}

    # Use the Bayesian model for predictions
    inference = VariableElimination(model)
    prediction = inference.query(variables=['stroke'], evidence=data_dict)

    # Extract the probability of stroke=1
    stroke_prob = prediction.values[1]
    return stroke_prob


if __name__ == '__main__':
    app.run(port=5000, debug=True)
