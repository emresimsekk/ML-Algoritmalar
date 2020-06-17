import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Verilerin Yüklenmesi
data=pd.read_csv('satislar.csv')

moon=data[['Aylar']] #data.iloc[:,0:1].values 
selling=data[['Satislar']] #data.iloc[:,1:2].values


#Verileri eğitim ve test için parçaladık
from sklearn.model_selection  import train_test_split
x_train,x_test,y_train,y_test=train_test_split(moon,selling,test_size=0.33,random_state=0)
'''
#verilerin standartlaşma ölçekleme yapıldı 
from sklearn.preprocessing import StandardScaler
sc=StandardScaler() 
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)
Y_train=sc.fit_transform(y_train)
Y_test=sc.fit_transform(y_test)
'''
# Model inşası (Linear Regression)
from sklearn.linear_model import LinearRegression
lr=LinearRegression();
lr.fit(x_train,y_train)
result=lr.predict(x_test)

# İndex Numarasına göre sırala
x_train=x_train.sort_index()
y_train=y_train.sort_index()
 

#Görselleştirme
plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))




