import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Verilerin Yüklenmesi
data=pd.read_csv('veriler.csv')

x=data.iloc[:,1:4].values #Bağımsız değişkenler
y=data.iloc[:,4:].values #Bağımlı değişkenler


#Verileri eğitim ve test için parçaladık
from sklearn.model_selection  import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=0)

#verilerin standartlaşma yapıldı 
from sklearn.preprocessing import StandardScaler
sc=StandardScaler() 
# fit=eğit, transform=eğitimi uygulama kullanma
X_train=sc.fit_transform(x_train)
X_test=sc.transform(x_test)

#Support Vector Machine 
from sklearn.svm import SVC
svc=SVC(kernel='rbf') #linear
svc.fit(X_train,y_train)

y_pred=svc.predict(X_test)
print(y_pred)
print(y_test)


# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)


