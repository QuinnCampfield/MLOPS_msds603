import requests
import pandas as pd
import numpy as np
import random

# Load a sample of the data
df = pd.read_csv("src/IncomeSurveyDataset.csv", nrows=10)

# Drop the target variable and any columns that were dropped in training
df = df.drop(['CONDMP', 'RENTM', 'Self_emp_income.1', 'Total_income'], axis=1)

# Create a few test cases from the data
test_cases = []
for i in range(3):
    # Get a random row
    row = df.iloc[random.randint(0, len(df)-1)]
    # Convert to list of features
    features = row.values.tolist()
    test_cases.append(features)

# Test the API with each test case
url = 'http://127.0.0.1:8000/predict'

print("Testing the Income Prediction API:")
print("-" * 50)

for i, features in enumerate(test_cases):
    print(f"\nTest Case {i+1}:")
    print(f"Features: {features}")
    
    # Make the API request
    response = requests.post(url, json={"features": features})
    
    # Print the result
    if response.status_code == 200:
        result = response.json()
        print(f"Predicted Income: ${result['predicted_income']:,.2f}")
    else:
        print(f"Error: {response.text}")
    
    print("-" * 50)