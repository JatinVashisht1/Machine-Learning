# Train a logistic regression classifier to predict whether a flower is iris verginica or not

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
# print(list(iris.keys())) # This will print all keys in iris dataset
# print(iris['data']) # print data in 'data' key of iris dataset  
# print(iris['DESCR']) # Print description of dataset
# print(iris['data'].shape)
'''
    #    Iris-Setosa      0
    #   Iris-Versicolour  1
    #   Iris-Virginica    2
'''

X = iris['data'][:, 3:] # this slicing means, load all rows, and load columns from 3rd column to all (column indexing also starts from 0)
# print(X)
Y = (iris['target'] == 2).astype(int) 
# This will return true or false, because we are trying to make a classifier to predict if flower is virginica or not
# but logistic regression will work with numbers not true/false, so we convert Y into int
# print(Y)  

clf = LogisticRegression()
clf.fit(X, Y)

example = clf.predict([[9.6]]) # perdict function takes a 2d array
print(example)

# Using MATPLOTLIB to plot the visualization

X_new = np.linspace(0,3,1000).reshape(-1,1) # linspace Return evenly spaced numbers over a specified interval and reshape will reshape the array
# X_new = np.linspace(0,3,1000).reshape(1,-1) # in contrast to -1, 1 it will provide data into a lot of columns
# print(X_new)

Y_prob = clf.predict_proba(X_new) # predict function used to take threshold of 0.5 probability and it used to tell whether the flower is verginica or not
# but predict_proba will return the actual value of probability


plt.plot(X_new, Y_prob[:, 1], "-g", label = "virginica") # second parameter is giving Y_prob value and sliced to give value of first row only
plt.show()
