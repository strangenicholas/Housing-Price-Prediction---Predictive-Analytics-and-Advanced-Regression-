import pickle
import numpy as np



# Load the Linear Regression model from the file
with open('xgboost_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Features
# 1450	5	896	1.0	730.0	882.0	896	1	5	1961
# OverallQual 
# GrLivArea 
# GarageCars
# GarageArea
# TotalBsmtSF
# 1stFlrSF
# FullBath
# TotRmsAbvGrd
# YearBuilt

# input = np.array([5,896,1.0,730.0,882.0,896,1,5,1961])
# input = input.reshape(1, -1)
# input = scaler.fit_transform(input)
# input = scaler.transform(input)
# print(input)

# Use the model to make predictions
X_test = np.array([5,896,1.0,730.0,882.0,896,1,5,1961,1961])
X_test = X_test.reshape(1, -1)

y_pred = model.predict(X_test)

# # Print the predicted values
print(y_pred)


