# -*- coding: utf-8 -*-
"""
Created on Wed May  2 21:10:59 2018

@author: akash
"""
# Machine Learning
# Data Processing Templates

import numpy as np
import matplotlib.pyplot as pl
import pandas as pd

# Importing Dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

# Spliting dataset in Test and Training set
from sklearn.cross_validation import train_test_split
X_train, X_test, T_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

#Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""￼

# Fitting Simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, T_train)

# predicting Test set results
Y_pred = regressor.predict(X_test)

# Visualising the training set results
pl.scatter(X_train, T_train, color='red')
pl.plot(X_train, regressor.predict(X_train), color = 'green')
pl.title('Salary Vs Experiance (Training Set)')
pl.xlabel('Years of Experiance')
pl.ylabel('Salary')
pl.show()

# Visualising the test set results
pl.scatter(X_test, Y_test, color='red')
pl.plot(X_train, regressor.predict(X_train), color = 'green')
pl.title('Salary Vs Experiance (Test Set)')
pl.xlabel('Years of Experiance')
pl.ylabel('Salary')
pl.show()