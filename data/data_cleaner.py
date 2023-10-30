import pandas as pd


# Function to categorize avg_glucose_level
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


# Load the dataset
data = pd.read_csv('raw.csv')

# Drop the entire rows with 'N/A' in 'bmi' column
data.dropna(subset=['bmi'], inplace=True)

# make sure the gender col is int
data['gender'] = data['gender'].map({'Male': 0, 'Female': 1})
data.dropna(subset=['gender'], inplace=True)
data['gender'] = data['gender'].astype(int)
data['ever_married'] = data['ever_married'].map({'Yes': 1, 'No': 0})
data['isPrivate'] = data['work_type'].apply(lambda x: 1 if x == 'Private' else 0)
data['isSelfEmployed'] = data['work_type'].apply(lambda x: 1 if x == 'Self-employed' else 0)

# Drop the original 'work_type' column
data.drop(columns=['work_type'], inplace=True)
data['Residence_type'] = data['Residence_type'].map({'Urban': 1, 'Rural': 0})

# Remove rows with 'Unknown' in 'smoking_status' column
data = data[data['smoking_status'] != 'Unknown']

# Encode 'smoking_status' column
smoking_mapping = {
    'never smoked': 0,
    'formerly smoked': 1,
    'smokes': 2
}
data['smoking_status'] = data['smoking_status'].map(smoking_mapping)

# Categorize 'avg_glucose_level' and 'bmi' columns
data['avg_glucose_level'] = data['avg_glucose_level'].apply(categorize_glucose_level)
data['bmi'] = data['bmi'].apply(categorize_bmi)
data['age'] = data['age'].apply(categorize_age)

# Ensure 'stroke' is the last column
cols = [col for col in data if col != 'stroke'] + ['stroke']
data = data[cols]

# drop the 'id' column
data.drop(columns=['id'], inplace=True)

# duplicate the rows with 'stroke' = 1 10 times to balance the dataset
stroke_data = data[data['stroke'] == 1]
data = data._append([stroke_data] * 17, ignore_index=True)

# set seed to 0 to ensure reproducibility
data = data.sample(frac=1, random_state=0).reset_index(drop=True)

# count the number of rows with 'stroke' = 1 and 'stroke' = 0
print(data['stroke'].value_counts())

# Save the transformed data to a CSV file
data.to_csv('transformed_data.csv', index=False)

# Display a message to indicate successful saving
print("Transformed data saved to 'transformed_data.csv'.")
