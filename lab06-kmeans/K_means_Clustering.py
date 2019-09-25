# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 10:34:51 2018
K-Means Clustering
@author: RoKu
"""
#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import dataset
dataset = pd.read_csv('Mall_Customers_v.csv')
X = dataset.iloc[:,[3,4]].values
"""
Here we are not predicting anything so there is no X and y here we only have X to clus
plot also we are not using certain columns of the dataset
No encoding needed
No splitting needed
"""
#Using elbow method to find optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',n_init=10,max_iter=300,random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

#use plot method to plot score and cluster
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel = ('Number of cluster')
plt.ylabel = ('wcss-values')

"""
Use the number of cluster such that the slope is less for further K and
is changing insignificantly...
Observe the 5-6 slope on this graph..
here 5 is the optimal value of the clusters
"""
##fitting the K-Means to the dataset
kmeans = KMeans(n_clusters=5,init='k-means++',n_init=10,max_iter=300,random_state=42)
y_kmeans = kmeans.fit_predict(X)
"""
here the fitting method is used with fit-pedict |^|^
"""
#visualize
"""
get to know how these clusters are stored in y_kmeans
""" 
plt.figure()
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c='red',  label='Standard')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c='blue', label='Target-1')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c='green',label='Exclude')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100,c='cyan', label='Careless-lure them')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=100,c='black',label='They will spend..target')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='yellow',label='Centroids')
plt.title('Clusters of customers')
plt.xlabel = ('Annual Income (in thousand rupees)')
plt.ylabel = ('Spending Score (range-1:100)')
plt.legend()
plt.show()


