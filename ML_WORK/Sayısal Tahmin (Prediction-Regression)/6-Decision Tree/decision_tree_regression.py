import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures

#Verilerin Yüklenmesi
data=pd.read_csv('maaslar.csv')

x=data.iloc[:,1:2]
y=data.iloc[:,2:]
X=x.values
Y=y.values


#Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor

dt_reg=DecisionTreeRegressor(random_state=0)
dt_reg.fit(X,Y)

#Grafikler Oluşturulur
plt.scatter(X,Y,color='red')
plt.plot(X,dt_reg.predict(X),color='blue')

#tahmin
print(dt_reg.predict(X))









