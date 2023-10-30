import logging
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the data
data = pd.read_csv('../../data/transformed_data.csv')

# Use the first 75% of the data for training
data_subset = data.iloc[:int(len(data) * 0.8)].copy()

# Adjust the edges to match dataset columns, removing references to "Lifestyle"
edges = [
    # Direct influences on stroke
    ("age", "stroke"),
    ("hypertension", "stroke"),
    ("heart_disease", "stroke"),

    # Gender influences various factors
    ("gender", "smoking_status"),
    ("gender", "isPrivate"),
    ("gender", "bmi"),
    ("gender", "stroke"),

    # Marital and residence factors
    ("ever_married", "Residence_type"),
    ("Residence_type", "stroke"),

    # Employment type based on age and marital status
    ("age", "isSelfEmployed"),
    ("ever_married", "isSelfEmployed"),
    ("isSelfEmployed", "stroke"),

    # Private insurance can be influenced by employment type and age
    ("isSelfEmployed", "isPrivate"),
    ("age", "isPrivate"),
    ("isPrivate", "stroke"),

    # Smoking status influences hypertension and heart disease
    ("smoking_status", "hypertension"),
    ("smoking_status", "heart_disease"),

    # BMI also influences hypertension and heart disease
    ("bmi", "hypertension"),
    ("bmi", "heart_disease"),

    # Glucose level might influence bmi and hypertension
    ("avg_glucose_level", "bmi"),
    ("avg_glucose_level", "hypertension")
]

# Define the Bayesian model structure based on the provided edges
model = BayesianNetwork(edges)

# Train the model using Maximum Likelihood Estimators on the subset data
model.fit(data_subset, estimator=MaximumLikelihoodEstimator)

# Display the learned CPDs for a sample node (e.g., "stroke")
print(model.get_cpds("stroke"))

with open("saved_complicated_model.pkl", "wb") as file:
    pickle.dump(model, file)
