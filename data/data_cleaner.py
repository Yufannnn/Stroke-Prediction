import pandas as pd

# Load the dataset
data = pd.read_csv('raw.csv')

# Display the first few rows for reference
print("Initial Data:")
print(data.head(10))
print("\n")

# Remove rows with 'Unknown' in 'smoking_status' column
data = data[data['smoking_status'] != 'Unknown']

# Encode 'smoking_status' column
smoking_mapping = {
    'never smoked': 0,
    'formerly smoked': 1,
    'smokes': 2
}
data['smoking_status'] = data['smoking_status'].map(smoking_mapping)

# Drop rows with 'N/A' in 'bmi' column
data = data[data['bmi'] != 'N/A']

# # Drop the 'id' column
# data.drop(columns=['id'], inplace=True)

# Convert 'ever_married' column to binary (1 for 'Yes', 0 for 'No')
data['ever_married'] = data['ever_married'].map({'Yes': 1, 'No': 0})

# Create binary columns for 'isPrivate' and 'isSelfEmployed'
data['isPrivate'] = data['work_type'].apply(lambda x: 1 if x == 'Private' else 0)
data['isSelfEmployed'] = data['work_type'].apply(lambda x: 1 if x == 'Self-employed' else 0)

# Drop the original 'work_type' column
data.drop(columns=['work_type'], inplace=True)

# Encode 'Residence_type' column
data['Residence_type'] = data['Residence_type'].map({'Urban': 1, 'Rural': 0})

# Ensure 'stroke' is the last column
cols = [col for col in data if col != 'stroke'] + ['stroke']
data = data[cols]

# Save the transformed data to a CSV file
data.to_csv('transformed_data.csv', index=False)

# Display a message to indicate successful saving
print("Transformed data saved to 'transformed_data.csv'.")
