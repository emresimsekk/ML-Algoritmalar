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


#verilerin Ölçeklendirme standartlaşma yapıldı 
from sklearn.preprocessing import StandardScaler

scalerX=StandardScaler() 
x_scaler=scalerX.fit_transform(X)

scalerY=StandardScaler() 
y_scaler=scalerY.fit_transform(Y)


#Support Vector Regression
from sklearn.svm import SVR

svr_reg=SVR(kernel='rbf')
svr_reg.fit(x_scaler,y_scaler)

#Grafikler Oluşturulur
plt.scatter(x_scaler,y_scaler,color='red')
plt.plot(x_scaler,svr_reg.predict(x_scaler),color='blue')










