import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('musteriler.csv')

X=data.iloc[:,2:4].values

#K-Means Algoritması
from sklearn.cluster import KMeans

kmeans=KMeans(n_clusters=3,init='k-means++')
kmeans.fit(X)

# 3 tane Merkez Noktası Belirledi
print(kmeans.cluster_centers_)
result=[]
for i in range(1,10):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=123)
    kmeans.fit(X)
    result.append(kmeans.inertia_)
    
plt.plot(range(1,10),result)