import numpy as np
import shap
import pickle
import pandas as pd
from pgmpy.inference import VariableElimination

# Load the saved model
with open("saved_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

# Print states of each node in the Bayesian Network
for node in loaded_model.nodes():
    print(f"Node: {node}")
    print(f"States: {loaded_model.get_cpds(node).state_names[node]}")
    print("-----")

# Load the dataset
data = pd.read_csv('../data/transformed_data.csv')

# Split the data into train and test
X = data.drop('stroke', axis=1)
y = data['stroke']


# Define a prediction function for the Bayesian Network
def bn_predict(data_array):
    # Convert numpy array to DataFrame if necessary
    if isinstance(data_array, pd.DataFrame):
        data = data_array
    else:
        data = pd.DataFrame(data_array, columns=X.columns)

    # Replace non-finite values with appropriate defaults
    data.fillna(0, inplace=True)  # fill NaN values with 0
    data.replace([np.inf, -np.inf], 0, inplace=True)  # replace infinite values with 0

    # Round the perturbed values to the nearest known category
    data = data.round().astype(int)

    predictions = []
    inference = VariableElimination(loaded_model)
    for _, row in data.iterrows():
        prediction = inference.map_query(variables=['stroke'], evidence=row.to_dict())
        predictions.append(prediction['stroke'])

    return np.array(predictions).reshape(-1, 1)


# Use KernelExplainer for models like Bayesian Networks
explainer = shap.KernelExplainer(bn_predict, X)

# Choose a sample to explain
i = 23
sample = X.iloc[i]

# Compute SHAP values
shap_values = explainer.shap_values(sample)

# Visualize the explanation
shap.initjs()
shap.force_plot(explainer.expected_value[0], shap_values[0], sample)
