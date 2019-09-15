# -*- coding: utf-8 -*-
"""
Created om Mon 02-Jul-2018
Multiple linear Regression
@Author-RoKu 
"""
'''
#step 1
#import libraries
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
#step 2
#import the dataset
'''

dataset = pd.read_csv('dataset.csv')
dataset = dataset.drop_duplicates()
#get your own variables
X =  dataset.iloc[:,:-1].values
y  = dataset.iloc[:,4].values

"""
here the column four is text entries(categorical data0 so it is not possible to extract and give weights 
so this type of data must be converted to the numbers so as to perform operations on it...and
thus data encoding must be used here 
"""
'''
#Step 3
##Encode the categorical data
#for this the scikit library is used
'''
from sklearn.preprocessing import LabelEncoder #we don't need onehotencoder as we don't need binary in output
labelencoder = LabelEncoder() #this is the object of class ->LabelEncoder
y = labelencoder.fit_transform(y) #the data is encodeddd can check X
#onehotencoder = OneHotEncoder(categorical_features=[3]) #this enables the onehotencoding for only [3] column
#X = onehotencoder.fit_transform(X).toarray() ##this makes three more columns and are encoded according tho city
# How to elimnate data
#Avoiding the unwanted input columns |||

'''
#Step4
#Split the data-set
'''

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 0)

'''
#step5
#fittng the MLR to the training set
'''
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)#,sample_weight=None)


'''
#step6
#Predicting the test set results ik
'''
y_pred = regressor.predict(x_test)
y_pred = abs(np.round(y_pred))
'''
'''
test_accuracy = regressor.score(x_test,y_test)
train_accuracy = regressor.score(x_train,y_train)

'''
#step7
#Building the Optimal Model using backward elimination
!**!
backward elimination method using the p value
"""
'''
#import stats model for statistical operations
import statsmodels.formula.api as sm
arr = np.ones((len(X),1))
X_opt = np.append(arr,values = X ,axis=1) 

##we appended a array of ones as A0 = 1 as it has the base weight
#Building the Optimal Model using backward elimination actuall

X_opt2 = X_opt[:,[1,3,4]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt2).fit() #ordinary least square (y-y^)^2 method of minimizing
regressor_OLS.summary()



from sklearn.metrics import confusion_matrix 
print(confusion_matrix(y_test,y_pred))

from sklearn.metrics import cohen_kappa_score as kappa
print(kappa(y_pred,y_test))
   