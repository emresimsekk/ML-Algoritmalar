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

#Polynomal Regression Dönüşüm Yapılıyor
poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(X) 

from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x_poly,y)

#Grafiler Çizdiriliyor
plt.scatter(X,Y, color='red')
#Linear regresyon içerisinde polinomal reg  x değerini verdik
plt.plot(X,lin_reg.predict(poly_reg.fit_transform(X)), color='blue')

#tahminler
print(lin_reg.predict(poly_reg.fit_transform(X)))











