import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# Load the data
df = pd.read_csv('framingham.csv')

# Display info and description of the data
print(df.info())
print(df.describe())

# Display value counts of the target variable
print(df['TenYearCHD'].value_counts())

# Split the data into features and target
X = df.drop('TenYearCHD', axis=1)
y = df['TenYearCHD']

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

# Handle NaN values by imputing missing values
imputer = SimpleImputer(strategy='mean')
x_train_imputed = imputer.fit_transform(x_train)
x_test_imputed = imputer.transform(x_test)  # Apply the same transformation to the test set

# Initialize the model
model = LogisticRegression(max_iter=1000)  # Increase max_iter if convergence is an issue

# Train the model with the imputed data
model.fit(x_train_imputed, y_train)

# Make predictions on the imputed test set
y_pred = model.predict(x_test_imputed)

# Calculate accuracy
score = accuracy_score(y_test, y_pred)
print("The accuracy score is", score)

# Prompt user for input for each feature
input_dict = {}
for column in X.columns:
    value = input(f"Enter value for {column} (leave blank for missing value): ")
    input_dict[column] = float(value) if value else np.nan

# Convert the input dictionary to DataFrame
input_df = pd.DataFrame(input_dict, index=[0])

# Handle NaN values in the input
input_imputed = imputer.transform(input_df)

# Predict on the new input
prediction = model.predict(input_imputed)
prediction_proba = model.predict_proba(input_imputed)

# Display prediction result
print("Prediction (0 - No CHD, 1 - CHD):", prediction[0])
print("Prediction Probability (0 - No CHD, 1 - CHD):", prediction_proba[0])
