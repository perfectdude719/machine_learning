#import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV,  PredefinedSplit
from sklearn.linear_model import SGDRegressor #for stochastic regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge #batch
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Path to a file in Google Drive
data = pd.read_csv('https://drive.google.com/uc?id=1nqSQ-M_Ff2TDD2U0syAbFBvJZuXJLWNc')
#data=pd.read_csv('California_Houses.csv')



# Define feature columns
X_features = [
    'Median_Income', 'Median_Age', 'Tot_Rooms', 'Tot_Bedrooms',
    'Population', 'Households', 'Latitude', 'Longitude',
    'Distance_to_coast', 'Distance_to_LA', 'Distance_to_SanDiego',
    'Distance_to_SanJose', 'Distance_to_SanFrancisco'
]

# Arrange data and split
X = data[X_features]  # Feature matrix
y = data['Median_House_Value']  # Target variable

# Transform to numpy arrays
X = X.to_numpy()  # or use X.values
y = np.array(y)

# Normalize the data for comparison later on
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

# Split the data into 70% training, 15% test, and 15% validation
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Split the normalized set as well
Xnorm_train, Xnorm_temp = train_test_split(X_norm, test_size=0.3, random_state=42)
Xnorm_val, Xnorm_test = train_test_split(Xnorm_temp, test_size=0.5, random_state=42)

# Visualize the data
print(f"The features of the first sample are: {X_train[1]}")
print(f"The median price is: {y_train[1]}\n")

print(f"The features of the first sample from the normalized set are: {Xnorm_train[1]}")
print(f"The median price is: {y_train[1]}")


# Initialize models
sgdr = SGDRegressor(max_iter=2000)
linear_model = LinearRegression()
lasso_model = Lasso(alpha=0.1)
ridge_model = Ridge(alpha=1)

# train and evaluate sgdr
sgdr.fit(Xnorm_train, y_train)
y_pred_sgdr=sgdr.predict(Xnorm_val)
mse_sgdr = mean_squared_error(y_val, y_pred_sgdr)
mae_sgdr = mean_absolute_error(y_val,y_pred_sgdr)

# Train and evaluate Linear Regression
linear_model.fit(Xnorm_train, y_train)
y_pred_linear = linear_model.predict(Xnorm_val)
mse_linear = mean_squared_error(y_val, y_pred_linear)
mae_linear = mean_absolute_error(y_val, y_pred_linear)

# Train and evaluate Lasso Regression
lasso_model.fit(Xnorm_train, y_train)
y_pred_lasso = lasso_model.predict(Xnorm_val)
mse_lasso = mean_squared_error(y_val, y_pred_lasso)
mae_lasso = mean_absolute_error(y_val, y_pred_lasso)

# Train and evaluate Ridge Regression
ridge_model.fit(Xnorm_train, y_train)
y_pred_ridge = ridge_model.predict(Xnorm_val)
mse_ridge = mean_squared_error(y_val, y_pred_ridge)
mae_ridge = mean_absolute_error(y_val, y_pred_ridge)

# Print the results
print("SGDR:")
print("MSE:", mse_sgdr)
print("MAE:", mae_sgdr)

print("\nLinear Regression:")
print("MSE:", mse_linear)
print("MAE:", mae_linear)

print("\nLasso Regression:")
print("MSE:", mse_lasso)
print("MAE:", mae_lasso)
print("Selected features:", sum(lasso_model.coef_ != 0))

print("\nRidge Regression:")
print("MSE:", mse_ridge)
print("MAE:", mae_ridge)


import numpy as np
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error

# Define a range for alpha
alpha_range = np.logspace(-4, 2, 10)

# Create scorers for MAE and MSE
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)  # MAE scorer
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)    # MSE scorer

# Combine the training and validation sets
X_combined = np.concatenate([Xnorm_train, Xnorm_val])
y_combined = np.concatenate([y_train, y_val])

# Define the parameter grid using alpha_range
param_grid = {'alpha': alpha_range}

# Initialize models
model_lasso = Lasso(max_iter=10000)
model_ridge = Ridge(max_iter=10000)

# Initialize GridSearchCV for Ridge and Lasso regression with multiple scorers
grid_search_ridge = GridSearchCV(model_ridge, param_grid, scoring={'MSE': mse_scorer, 'MAE': mae_scorer}, cv=5, refit='MSE')
grid_search_lasso = GridSearchCV(model_lasso, param_grid, scoring={'MSE': mse_scorer, 'MAE': mae_scorer}, cv=5, refit='MSE')

# Fit using the combined dataset
grid_search_ridge.fit(X_combined, y_combined)
grid_search_lasso.fit(X_combined, y_combined)

# Best score and best parameters for Ridge
print("Ridge Regression:")
print("Best score (MSE):", -grid_search_ridge.best_score_)  # Negate to show as positive
print("Best parameters:", grid_search_ridge.best_params_)
print("Best MAE score:", -grid_search_ridge.cv_results_['mean_test_MAE'][grid_search_ridge.best_index_])  # Negate to show as positive

# Best score and best parameters for Lasso
print("\nLasso Regression:")
print("Best score (MSE):", -grid_search_lasso.best_score_)  # Negate to show as positive
print("Best parameters:", grid_search_lasso.best_params_)
print("Best MAE score:", -grid_search_lasso.cv_results_['mean_test_MAE'][grid_search_lasso.best_index_])  # Negate to show as 

#retrain the model with the parametr that scored best
best_ridge_model = grid_search_ridge.best_estimator_
best_lasso_model = grid_search_lasso.best_estimator_

# Make predictions on the test set
y_pred_ridge = best_ridge_model.predict(Xnorm_test)
y_pred_lasso = best_lasso_model.predict(Xnorm_test)

# Evaluate the Ridge model on the test set
print("Ridge Regression Test Set Evaluation:")
print("MAE:", mean_absolute_error(y_test, y_pred_ridge))
print("MSE:", mean_squared_error(y_test, y_pred_ridge))
print("R²:", r2_score(y_test, y_pred_ridge))

# Evaluate the Lasso model on the test set
print("\nLasso Regression Test Set Evaluation:")
print("MAE:", mean_absolute_error(y_test, y_pred_lasso))
print("MSE:", mean_squared_error(y_test, y_pred_lasso))
print("R²:", r2_score(y_test, y_pred_lasso))



# Scatter plots for each feature
features = X_features  # List of feature names
models = {'Linear Regression': y_pred_linear, 'Lasso Regression': y_pred_lasso, 'Ridge Regression': y_pred_ridge}

for model_name, y_pred in models.items():
    print(f"Scatter plots for {model_name}")
    plt.figure(figsize=(20, 15))
    
    for idx, feature in enumerate(features, 1):
        plt.subplot(4, 4, idx)
        plt.scatter(X_test[:, idx-1], y_test, color='blue', label='True Value', alpha=0.6)
        plt.scatter(X_test[:, idx-1], y_pred, color='red', label='Predicted Value', alpha=0.6)
        plt.xlabel(feature)
        plt.ylabel('Median House Value')
        plt.title(f"{feature} vs. True & Predicted Values")
        if idx == 1:
            plt.legend()
    
    plt.tight_layout()
    plt.show()
