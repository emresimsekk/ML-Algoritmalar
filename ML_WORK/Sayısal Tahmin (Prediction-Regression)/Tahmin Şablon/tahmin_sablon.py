import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

#Verilerin Yüklenmesi
data=pd.read_csv('maas.csv')

x=data.iloc[:,2:5]
y=data.iloc[:,5:6]
X=x.values
Y=y.values

# Linear Regression Model
from sklearn.linear_model import LinearRegression
ln_reg=LinearRegression()
ln_reg.fit(X,Y)

#Linear Regression P-Value
model = sm.OLS(ln_reg.predict(X),X )
results = model.fit().summary()
print(results)

#----------------------------------------------------------------------------------

# Polynomial Regression Model
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(X) 
lin_reg=LinearRegression()
lin_reg.fit(x_poly,y)

#Polynomial Regression P-Value
model = sm.OLS(lin_reg.predict(poly_reg.fit_transform(X)),X )
results = model.fit().summary()
print(results)

#----------------------------------------------------------------------------------

#Support Vector Regression
#verilerin Ölçeklendirme standartlaşma yapıldı 
from sklearn.preprocessing import StandardScaler
scalerX=StandardScaler() 
x_scaler=scalerX.fit_transform(X)
scalerY=StandardScaler() 
y_scaler=scalerY.fit_transform(Y)

from sklearn.svm import SVR
svr_reg=SVR(kernel='rbf')
svr_reg.fit(x_scaler,y_scaler)

#Support Vector Regression Regression P-Value
model = sm.OLS(svr_reg.predict(x_scaler),x_scaler )
results = model.fit().summary()
print(results)

#----------------------------------------------------------------------------------

#Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
dt_reg=DecisionTreeRegressor(random_state=0)
dt_reg.fit(X,Y)

#Decision Tree Regression P-Value
model = sm.OLS(dt_reg.predict(X),X )
results = model.fit().summary()
print(results)

#----------------------------------------------------------------------------------

#Random Forest Regression n_estimators=10 demek kaç tane ağaç oluşturacak
from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators=10,random_state=0)
rf_reg.fit(X,Y)

#Random Forest Regression P-Value
model = sm.OLS(rf_reg.predict(X),X )
results = model.fit().summary()
print(results)

#----------------------------------------------------------------------------------

#r2_score Değerleri
from sklearn.metrics import r2_score
print('r2_score Değeri :')
print(r2_score(Y,rf_reg.predict(X)))

#----------------------------------------------------------------------------------

#Eksik Verileri Ortalama ile Doldurma İmpuuter
from sklearn.preprocessing import Imputer

imputer=Imputer(missing_values='NaN',strategy='mean',axis=0) #param1=hatalı değer, param2=ortalama, param3=satır bazlı
sizeWeightAge=data.iloc[:,1:4].values #iloc ile veri çekme
imputer=imputer.fit(sizeWeightAge[:,1:4])
sizeWeightAge[:,1:4]=imputer.transform(sizeWeightAge[:,1:4])

#----------------------------------------------------------------------------------

#Encoder Nomimal Ordinal->Numeric Kategorileri 0 1 mantığına dönüştürüyor
country=data.iloc[:,0:1].values
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
country[:,0]=le.fit_transform(country[:,0])
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(categorical_features='all')
country=ohe.fit_transform(country).toarray()











