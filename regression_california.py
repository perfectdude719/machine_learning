# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Set a random seed for reproducibility
np.random.seed(42)
### Regression Task - California Housing Prices Data ###

# Step 1: Load the dataset
# Replace 'california_housing_data.csv' with your actual dataset path
housing_data = pd.read_csv('California_Houses.csv')

# Step 2: Split the data into features and target
X = housing_data.drop('Median_House_Value', axis=1)  # Features
y = housing_data['Median_House_Value']  # Target variable

# Step 3: Split dataset into training, validation, and testing sets (70%, 15%, 15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Step 4: Initialize models
linear_model = LinearRegression()
lasso_model = Lasso(alpha=0.1)
ridge_model = Ridge(alpha=0.1)

# Dictionary to store the performance of each model
regression_metrics = {}

# Step 5: Train and validate each model

## Linear Regression
linear_model.fit(X_train, y_train)
y_val_pred = linear_model.predict(X_val)
regression_metrics['Linear'] = {
    'MSE': mean_squared_error(y_val, y_val_pred),
    'MAE': mean_absolute_error(y_val, y_val_pred)
}

## Lasso Regression
lasso_model.fit(X_train, y_train)
y_val_pred = lasso_model.predict(X_val)
regression_metrics['Lasso'] = {
    'MSE': mean_squared_error(y_val, y_val_pred),
    'MAE': mean_absolute_error(y_val, y_val_pred)
}

## Ridge Regression
ridge_model.fit(X_train, y_train)
y_val_pred = ridge_model.predict(X_val)
regression_metrics['Ridge'] = {
    'MSE': mean_squared_error(y_val, y_val_pred),
    'MAE': mean_absolute_error(y_val, y_val_pred)
}

# Step 6: Report validation performance metrics for each model
print("Validation Set Performance:")
for model_name, metrics in regression_metrics.items():
    print(f"{model_name} Regression - MSE: {metrics['MSE']}, MAE: {metrics['MAE']}")

# Step 7: Choose the best model based on MSE and test it on the test set
best_model_name = min(regression_metrics, key=lambda x: regression_metrics[x]['MSE'])
if best_model_name == 'Linear':
    best_model = linear_model
elif best_model_name == 'Lasso':
    best_model = lasso_model
else:
    best_model = ridge_model

# Evaluate the best model on the test set
y_test_pred = best_model.predict(X_test)
print(f"\nBest Model ({best_model_name}) Test Set Performance:")
print("MSE:", mean_squared_error(y_test, y_test_pred))
print("MAE:", mean_absolute_error(y_test, y_test_pred))