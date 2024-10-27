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

### Classification Task - MAGIC Gamma Telescope Data ###

# Step 1: Load the dataset
# Replace 'magic_gamma_data.csv' with your actual dataset path
gamma_data = pd.read_csv('magic04.data')

# Step 2: Check class distribution to identify imbalance
print("Class distribution:\n", gamma_data['class'].value_counts())

# Step 3: Balance the classes
# Separate the two classes
gamma_class = gamma_data[gamma_data['class'] == 'g']
hadron_class = gamma_data[gamma_data['class'] == 'h']

# Downsample the gamma class to match hadron class size
gamma_class_downsampled = gamma_class.sample(len(hadron_class), random_state=42)

# Combine the balanced dataset
balanced_data = pd.concat([gamma_class_downsampled, hadron_class])

# Step 4: Split data into features and labels
X = balanced_data.drop('class', axis=1)  # Features
y = balanced_data['class']  # Labels

# Step 5: Split dataset into training, validation, and testing sets (70%, 15%, 15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Step 6: Initialize an empty dictionary to store performance metrics for each k
performance_metrics = {}

# Step 7: Iterate through different values of k to find the best one
for k in range(1, 11):
    # Initialize the K-NN classifier with the current k value
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Train the model on the training data
    knn.fit(X_train, y_train)
    
    # Validate the model on the validation data
    y_pred = knn.predict(X_val)
    
    # Calculate and store performance metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, pos_label='g')
    recall = recall_score(y_val, y_pred, pos_label='g')
    f1 = f1_score(y_val, y_pred, pos_label='g')
    conf_matrix = confusion_matrix(y_val, y_pred)
    
    # Store metrics in dictionary
    performance_metrics[k] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix
    }

# Step 8: Find the best k based on validation F1 score
best_k = max(performance_metrics, key=lambda k: performance_metrics[k]['f1_score'])
print(f"Best k found: {best_k}")

# Step 9: Retrain the best model on the training set and evaluate on the test set
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train, y_train)
y_test_pred = best_knn.predict(X_test)

# Step 10: Report test performance metrics
print("Test Set Performance:")
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print("Precision:", precision_score(y_test, y_test_pred, pos_label='g'))
print("Recall:", recall_score(y_test, y_test_pred, pos_label='g'))
print("F1 Score:", f1_score(y_test, y_test_pred, pos_label='g'))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))