# -*- coding: utf-8 -*-
"""
Created om Mon 02-Jul-2018
Simple linear Regression
@Author-RoKu 
"""
#step 1
#import libraries
import pandas as pd
import matplotlib.pyplot as plt

#step 2
#import dataset
dataset = pd.read_csv('Dataset_flwr.csv'); 
X = dataset.iloc[:,:-1]
y = dataset.iloc[:,1]

#step3
#divide the dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)

#step 4
#fit the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression();
regressor.fit(x_train,y_train)

#step 5
#predict on test dataset
y_pred = regressor.predict(x_test);
y_pred2 = regressor.predict(x_train);

#miscellenous
#accuracy
accuracy = regressor.score(x_test,y_test)
#s/tep 6
#visualizing the trained set result #run both together for getting combined
plt.scatter(x_train,y_train,color='red') #plots the visual result
plt.plot(x_train,regressor.predict(x_train),color='blue')  ##plots the model line
plt.title('Sepal Length V/S Petal Length for training set')
plt.xlabel('Petal Length')
plt.ylabel('Sepal Length')
plt.show()

#visualizing the test set result #run both together for getting combined
plt.scatter(x_test,y_test,color='red') #plots the visual result
plt.plot(x_train,regressor.predict(x_train),color='blue')  ##this is same as object regressor is the same 
#NO need to train again |^^
plt.title('Sepal Length V/S Petal Length for training set')
plt.xlabel('Petal Length')
plt.ylabel('Sepal Length')
plt.show()