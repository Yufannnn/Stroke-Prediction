from flask import Flask, request, jsonify
from pgmpy.inference import VariableElimination
import pickle
import os

app = Flask(__name__)

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
    gender = data_list[0]
    age = data_list[1]
    hypertension = data_list[2]
    heart_disease = data_list[3]
    ever_married = data_list[4]
    work_type = data_list[5]
    Residence_type = data_list[6]
    avg_glucose_level = data_list[7]
    bmi = data_list[8]
    smoking_status = data_list[9]

    # Gender validation and mapping
    if gender not in ['Male', 'Female']:
        raise ValueError("Invalid value for gender. It can be either 'Male' or 'Female'.")
    gender = {'Male': 0, 'Female': 1}[gender]

    # ever_married validation and mapping
    if ever_married not in ['Yes', 'No']:
        raise ValueError("Invalid value for ever_married. It can be either 'Yes' or 'No'.")
    ever_married = {'Yes': 1, 'No': 0}[ever_married]

    # work_type validation
    if work_type not in ['Private', 'Self-employed', 'Govt']:
        raise ValueError("Invalid work_type. Valid values are 'Private', 'Self-employed', 'Govt")
    isPrivate = 1 if work_type == 'Private' else 0
    isSelfEmployed = 1 if work_type == 'Self-employed' else 0

    # Residence_type validation and mapping
    if Residence_type not in ['Urban', 'Rural']:
        raise ValueError("Invalid value for Residence_type. It can be either 'Urban' or 'Rural'.")
    Residence_type = {'Urban': 1, 'Rural': 0}[Residence_type]

    # Smoking_status validation and mapping
    if smoking_status not in ['never smoked', 'formerly smoked', 'smokes', 'Unknown']:
        raise ValueError("Invalid value for smoking_status. Valid values are 'never smoked', 'formerly smoked', "
                         "'smokes', 'Unknown'.")
    if smoking_status == 'Unknown':
        raise ValueError("Invalid value for smoking_status: 'Unknown'")
    smoking_status = {'never smoked': 0, 'formerly smoked': 1, 'smokes': 2}[smoking_status]

    # Categorize avg_glucose_level, bmi, and age
    avg_glucose_level = categorize_glucose_level(avg_glucose_level)
    bmi = categorize_bmi(bmi)
    age = categorize_age(age)

    # Return the processed data as a list
    return [gender, age, hypertension, heart_disease, ever_married, isPrivate, isSelfEmployed, Residence_type,
            avg_glucose_level, bmi, smoking_status]


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from POST request
        data_list = request.get_json(force=True)

        # Preprocess the input data
        processed_data = preprocess_input_data(data_list)

        # Create a dictionary from the processed data for inference
        data_dict = {
            'gender': processed_data[0],
            'age': processed_data[1],
            'hypertension': processed_data[2],
            'heart_disease': processed_data[3],
            'ever_married': processed_data[4],
            'isPrivate': processed_data[5],
            'isSelfEmployed': processed_data[6],
            'Residence_type': processed_data[7],
            'avg_glucose_level': processed_data[8],
            'bmi': processed_data[9],
            'smoking_status': processed_data[10]
        }

        # Use the Bayesian model for predictions
        inference = VariableElimination(model)
        prediction = inference.query(variables=['stroke'], evidence=data_dict)

        # Extract the probability of stroke=1
        stroke_prob = prediction.values[1]

        # Return the result
        return jsonify({"stroke_probability": stroke_prob})

    except ValueError as e:
        # Return error message from preprocessing
        return jsonify({"error": str(e)}), 400  # 400 Bad Request


if __name__ == '__main__':
    app.run(port=5000, debug=True)
