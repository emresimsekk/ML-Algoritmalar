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


#Random Forest Regression n_estimators=10 demek kaç tane ağaç oluşturacak
from sklearn.ensemble import RandomForestRegressor

rf_reg=RandomForestRegressor(n_estimators=10,random_state=0)
rf_reg.fit(X,Y)

#Grafikler Oluşturulur
plt.scatter(X,Y,color='red')
plt.plot(X,rf_reg.predict(X),color='blue')

#tahmin
print(rf_reg.predict(X))










