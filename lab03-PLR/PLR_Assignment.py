"""
Problem Statement:
    Find the Item_Outlet_Sales for----> item weight = 10.85, item-vis = 0.01589,Item_MRP = 148
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 11:55:59 2018
Assignment on PLR 
@author: RoKu
"""
#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1]
y = dataset.iloc[:,1]

from sklearn.linear_model import LinearRegression
linReg = LinearRegression()

from sklearn.preprocessing import PolynomialFeatures
polyReg = PolynomialFeatures(degree=2)
X_poly = polyReg.fit_transform(X)  ###check the X_poly it is noe a0,a1 they are x1^0,x1^1,x1^2
polyReg.fit(X_poly,y)
linReg.fit(X_poly,y)

#Visualizing the polynomial regression results with higher resolution and smmother curve
plt.scatter(X, y, color='red')
plt.plot(X, linReg.predict(X_poly), color='blue')
plt.title('Sepal length Predictor')
plt.xlabel = ('Petal Length')
plt.ylabel = ('Species')
plt.grid()
plt.show()
"""
Predicting the sepla length for PetalLength=6.4
"""
arr = polyReg.fit_transform(6.4)
print(linReg.predict(arr))
linReg.score(X_poly,y)



