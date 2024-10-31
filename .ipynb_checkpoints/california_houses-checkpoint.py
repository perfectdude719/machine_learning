import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error

#read data#
data=pd.read_csv(California_Houses.csv)

#arrange data and split
X= data.drop('Median_House_Value',axis=1) #drop the y value --->feature matrix
y=data['Median_House_Value']#y value

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

#initialize models
linear_model= LinearRegression()
lasso_model=Lasso(alpha=0.1)
ridge_model=Ridge(alpha=0.1)

