# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 10:24:12 2019

@author: RoKu
"""

# Apriori
# Apriori
# Apriori

# Importing the libraries
import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  

# Data Preprocessing
store_data = pd.read_csv('store_data.csv',header=None)
transactions = []
for i in range(0, 7501):
   transactions.append([str(store_data.values[i,j]) for j in range(0,20)])

# Training Apriori on the dataset 
from apyori import apriori  
rules = apriori(transactions, min_support = 0.0045, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualising the results
results = list(rules)

#print
print(results)
