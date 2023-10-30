import pickle
import pandas as pd
from pgmpy.inference import VariableElimination

with open("saved_castrated_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

# 1. Prepare the Test Data
data = pd.read_csv('../../data/transformed_data.csv')

# use the last 25% of the data for testing
test_data = data.iloc[int(len(data) * 0.8):].copy()
test_data.drop(columns=['stroke'], inplace=True)  # Assuming 'stroke' is what you want to predict
# drop Residences, isSelfEmployed
test_data.drop(columns=['Residence_type', 'isSelfEmployed', 'isPrivate'], inplace=True)

# 2. Make Predictions
inference = VariableElimination(loaded_model)

# Loop through each row in test_data and predict
predicted_probabilities = []
for index, row in test_data.iterrows():
    # Get the probability distribution for 'stroke' given the evidence
    prediction = inference.query(variables=['stroke'], evidence=row.to_dict())
    # Store the probabilities of stroke=0 and stroke=1
    predicted_probabilities.append({
        'stroke=0': prediction.values[0],
        'stroke=1': prediction.values[1]
    })
actual_values = data.iloc[int(len(data) * 0.8):]['stroke'].tolist()

# print the predicted probabilities and the ground truth
print("Predicted Probabilities:", predicted_probabilities)
print("Actual Values:", data.iloc[int(len(data) * 0.8):]['stroke'].tolist())

# If you still want to evaluate accuracy using the most probable prediction
predicted_values = [1 if prob['stroke=1'] > prob['stroke=0'] else 0 for prob in predicted_probabilities]
correct_predictions = sum([1 for pred, actual in zip(predicted_values, actual_values) if pred == actual])
accuracy = correct_predictions / len(actual_values) * 100
print(f"Model Accuracy: {accuracy:.2f}%")

