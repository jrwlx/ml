import numpy as np
import pandas as pd
from sklearn import linear_model
import sklearn
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

# loading data
data = pd.read_csv("student-mat.csv", sep=";")

# print(data.head())

# trimming data
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# separating data
predict = 'G3'
# label is attribute to predict
# features is other attributes that will determine label
x = np.array(data.drop([predict], 1)) # array that contains features
y = np.array(data[predict]) # array that contains label

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# implementing linear regression
# defining model
linear = linear_model.LinearRegression()

# training models
linear.fit(x_train, y_train)
accuracy = linear.score(x_test, y_test)
# print(accuracy)

# saving model with pickle
with open("studentgrades.pickle", "wb") as f:
    pickle.dump(linear, f)

# viewing constants
# print('Coefficient: \n', linear.coef_) # slope values
# print('Intercept: \n', linear.intercept_) # intercept

predictions = linear.predict(x_test) # gets a list of all predictions
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
