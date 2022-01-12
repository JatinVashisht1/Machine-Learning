import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()

# print(diabetes.keys()) # {any-dataset}.keys will tell all keys of the {dataset}
# in this case they are: dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])
# print(diabetes.data) # To get the data inside the dataset
# print(diabetes.DESCR) # To get description of the dataset

# diabetes_X = diabetes.data[:, np.newaxis, 2] # this statement will slice the parent array and give an numpy array of array of feature at index 2
diabetes_X = diabetes.data # this statement will take all features
# print(diabetes_X) #print the array

diabetes_X_train = diabetes_X[:-30] 
diabetes_X_test = diabetes_X[-30:] # taking last 30 elements for testing
# For corresponding Y axis
diabetes_Y_train = diabetes.target[:-30]
diabetes_Y_test = diabetes.target[-30:]

model = linear_model.LinearRegression()

model.fit(diabetes_X_train, diabetes_Y_train) # We have created model here

diabetes_Y_predict =  model.predict(diabetes_X_test)

# Mean squared error is nothing but mean of sum of squared error (SSE)

print(f"mean of squared error is {mean_squared_error(diabetes_Y_test, diabetes_Y_predict)}")
print(f"weights: {model.coef_} and intercept: {model.intercept_}")


# below ploting will not work if we consider all features
# plt.scatter(diabetes_X_test, diabetes_Y_test)

# plt.plot(diabetes_X_test, diabetes_Y_predict)
# plt.show()

# previous mean squared error
# mean of squared error is 3035.0601152912695
# weights: [941.43097333] and intercept: 153.39713623331698