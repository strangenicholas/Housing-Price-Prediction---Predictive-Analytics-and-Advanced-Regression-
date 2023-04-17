import pickle
import numpy as np


# Load the Linear Regression model from the file
with open('linear_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Use the model to make predictions
X_test = np.array([0.44444444, 0.17066505, 0.25, 0.52517986, 0.3687291, 0.21918877, 0.33333333, 0.3, 0.64963504])
X_test = X_test.reshape(1, -1)

y_pred = model.predict(X_test)

# Print the predicted values
print(y_pred)

