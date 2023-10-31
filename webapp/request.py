import requests

def make_request(data):
    """Function to make a request and print the response."""
    response = requests.post(url, json=data)
    json_response = response.json()

    # Check if 'stroke_probability' is in the response (indicating a valid prediction)
    if 'stroke_probability' in json_response:
        print(f"Prediction for stroke probability: {json_response['stroke_probability']:.5f}")
    elif 'error' in json_response:  # If 'error' key exists, print the error message
        print(f"Error: {json_response['error']}")
    else:
        print("Unknown response:", json_response)

# Define the API endpoint
url = "http://127.0.0.1:5000/predict"

# Define a sample input data
valid_data = [
    "Male",       # gender
    67,           # age
    0,            # hypertension
    1,            # heart_disease
    "Yes",        # ever_married
    "Private",    # work_type
    "Urban",      # Residence_type
    228.69,       # avg_glucose_level
    36.6,         # bmi
    "formerly smoked"  # smoking_status
]

print("Sending valid data...")
make_request(valid_data)

invalid_data_gender = [
    "Non-binary",  # Invalid gender
    67,
    0,
    1,
    "Yes",
    "Private",
    "Urban",
    228.69,
    36.6,
    "formerly smoked"
]

print("\nSending data with invalid gender...")
make_request(invalid_data_gender)

invalid_data_work_type = [
    "Male",
    67,
    0,
    1,
    "Yes",
    "Freelancer",  # Invalid work type
    "Urban",
    228.69,
    36.6,
    "formerly smoked"
]

print("\nSending data with invalid work_type...")
make_request(invalid_data_work_type)

invalid_data_smoking_status = [
    "Male",
    67,
    0,
    1,
    "Yes",
    "Private",
    "Urban",
    228.69,
    36.6,
    "sometimes"  # Invalid smoking_status
]

print("\nSending data with invalid smoking status...")
make_request(invalid_data_smoking_status)
