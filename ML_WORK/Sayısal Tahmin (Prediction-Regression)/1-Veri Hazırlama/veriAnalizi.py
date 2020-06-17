import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Verilerin Yüklenmesi
data=pd.read_csv('veriler.csv')

#Veri Erişip Kullanımı
size=data[['boy']]
sizeWeight=data[['boy','kilo']]

#Eksik Verileri Ortalama ile Doldurma İmpuuter
from sklearn.preprocessing import Imputer

imputer=Imputer(missing_values='NaN',strategy='mean',axis=0) #param1=hatalı değer, param2=ortalama, param3=satır bazlı
sizeWeightAge=data.iloc[:,1:4].values #iloc ile veri çekme
imputer=imputer.fit(sizeWeightAge[:,1:4])
sizeWeightAge[:,1:4]=imputer.transform(sizeWeightAge[:,1:4])

#Encoder Nomimal Ordinal->Numeric Kategorileri 0 1 mantığına dönüştürüyor
country=data.iloc[:,0:1].values
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
country[:,0]=le.fit_transform(country[:,0])
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(categorical_features='all')
country=ohe.fit_transform(country).toarray()

#Dönüştürülen Sonuçları DataFrame toplama
resultCountry=pd.DataFrame(data=country, index=range(22), columns=['fr','tr','us'])
resultSizeWeightAge=pd.DataFrame(data=sizeWeightAge, index=range(22), columns=['Size','Weight','Age'])
gender=data.iloc[:,-1].values
resultGender=pd.DataFrame(data=gender, index=range(22), columns=['Gender'])
#Birleştirme İşlemi
s1=pd.concat([resultCountry,resultSizeWeightAge], axis=1)
s2=pd.concat([s1,resultGender], axis=1)

#Verileri eğitim ve test için parçaladık
from sklearn.model_selection  import train_test_split
x_train,x_test,y_train,y_test=train_test_split(s1,gender,test_size=0.33,random_state=0)

#verilerin standartlaşma yapıldı 
from sklearn.preprocessing import StandardScaler
sc=StandartScaler() 
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)



