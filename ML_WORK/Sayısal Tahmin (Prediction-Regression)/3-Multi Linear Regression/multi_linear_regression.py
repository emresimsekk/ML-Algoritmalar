import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Verilerin Yüklenmesi
data=pd.read_csv('veriler.csv')

sizeWeightAge=data.iloc[:,1:4]

#Encoder Nomimal Ordinal->Numeric Kategorileri 0 1 mantığına dönüştürüyor
#Ülke
country=data.iloc[:,0:1].values
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
country[:,0]=le.fit_transform(country[:,0])

from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(categorical_features='all')
country=ohe.fit_transform(country).toarray()

#Cinsiyet
gender=data.iloc[:,-1:].values
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
gender[:,0]=le.fit_transform(gender[:,0])


#Dönüştürülen Sonuçları DataFrame toplama kukla değişkene dikkat cinsiyette
resultCountry=pd.DataFrame(data=country, index=range(22), columns=['fr','tr','us'])
resultSwa=pd.DataFrame(data=sizeWeightAge, index=range(22), columns=['boy','kilo','yas'])
resultGender=pd.DataFrame(data=gender[:,0:1], index=range(22), columns=['cinsiyet'])

resultCSWA=pd.concat([resultCountry,resultSwa],axis=1)
resultFull=pd.concat([resultCSWA,resultGender],axis=1)


#Verileri eğitim ve test için parçaladık
from sklearn.model_selection  import train_test_split
x_train,x_test,y_train,y_test=train_test_split(resultCSWA,resultGender,test_size=0.33,random_state=0)


#----------------- Multible Linear Regression Cinsiyet -----------------
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
#veriyi eğit
regressor.fit(x_train,y_train)
#test et
y_pred=regressor.predict(x_test)


#----------------- Multible Linear Regression Boy -----------------
size=resultFull.iloc[:,3:4].values
left=resultFull.iloc[:,:3]
right=resultFull.iloc[:,4:]

newData=pd.concat([left,right],axis=1)

x_train,x_test,y_train,y_test=train_test_split(newData,size,test_size=0.33,random_state=0)

regressor2=LinearRegression()
#veriyi eğit
regressor2.fit(x_train,y_train)
#test et
y_pred=regressor2.predict(x_test)

#p value hesaplama ona göre backword yapılacak
import statsmodels.api as sm
X =np.append(arr = np.ones((22,1)).astype(int), values=newData ,axis=1 )

X_l = newData.iloc[:,[0,1,2,3]].values
#Dönüşümler yapıldı
X_l = X_l.astype(np.float64)
size=size.astype(np.float64)

model = sm.OLS(size,X_l )
results = model.fit()
print(results.summary())




