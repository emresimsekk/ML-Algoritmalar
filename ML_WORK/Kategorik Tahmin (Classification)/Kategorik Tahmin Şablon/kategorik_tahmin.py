import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Confusion Matrix
from sklearn.metrics import confusion_matrix

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

# Logistic Regression
from sklearn.linear_model import LogisticRegression
logr=LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred=logr.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
print('Logistic Regression')
print(cm)

#KNN 
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1,metric="minkowski")
knn.fit(X_train,y_train)

y_pred=knn.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
print('KNN ')
print(cm)

#Support Vector Machine 
from sklearn.svm import SVC
svc=SVC(kernel='rbf') #linear
svc.fit(X_train,y_train)

y_pred=svc.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
print('Support Vector Machine')
print(cm)

#Support Vector Machine Kernel Hack
from sklearn.svm import SVC
svc=SVC(kernel='poly') 
svc.fit(X_train,y_train)

y_pred=svc.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
print('Support Vector Machine Kernel Hack')
print(cm)

#Naive bayes 
from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB() 
gnb.fit(X_train,y_train)

y_pred=gnb.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
print('Naive Bayes')
print(cm)


#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier( criterion='entropy') 
dtc.fit(X_train,y_train)

y_pred=dtc.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
print('Decision Tree')
print(cm)

#Random Forest 
from sklearn.ensemble import RandomForestClassifier
rtc=RandomForestClassifier( n_estimators=10,criterion='entropy') 
rtc.fit(X_train,y_train)

y_pred=rtc.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
print('Random Forest ')
print(cm)

#Tahmin Olasılıkları
y_proba=rtc.predict_proba(X_test)
print(y_proba[:,0])

#False Pozitif True Pozitif 
from sklearn import metrics
fpr, tpr, thold=metrics.roc_curve(y_test,y_proba[:,0],pos_label='e')
print(fpr)
print(tpr)
