# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 10:11:04 2018
Polynomial Linear Regression
@author: RoKu
"""

"""
the cost function is a polynomial of various degrees of the same feature or multiple 
this helps us to get all the featuristic aspects of the feature by fitting not just the 
straight line but also the region
"""
"""
Linearity is the aspect of the coefficiants of the equation
"""
#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv('position_salary_plr.csv')
X = dataset.iloc[:,1:2]
y = dataset.iloc[:,-1]

##fitting the Linear regression model 
from sklearn.linear_model import LinearRegression  
linReg = LinearRegression()
linReg.fit(X,y)

from sklearn.preprocessing import PolynomialFeatures
polyReg = PolynomialFeatures(degree=4)
X_poly = polyReg.fit_transform(X)  ###check the X_poly it is noe a0,a1 they are x1^0,x1^1,x1^2
polyReg.fit(X_poly,y)
linreg1 = LinearRegression()
linreg1.fit(X_poly,y)

##visualizing the both fits
##Linear Regression
plt.scatter(X,y,color='red')
plt.plot(X,linReg.predict(X),color='blue')
plt.title('Bluff detector(linear regression)')
plt.xlabel = 'Position'
plt.ylabel = 'Salary'
plt.show()

##Polynomial Regression
plt.scatter(X, y, color='red')
plt.plot(X, linreg1.predict(X_poly), color='blue')
plt.title('Bluff detector(Polynomial linear regression)')
plt.xlabel = 'Position'
plt.ylabel = 'Salary'
plt.show()


##Predictig the result with Linear regression
print(linreg1.predict(polyReg.fit_transform(6.5)))
