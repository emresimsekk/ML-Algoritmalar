import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Verilerin Yüklenmesi
data=pd.read_csv('tenis.csv')


#Encoder Nomimal Ordinal->Numeric Kategorileri 0 1 mantığına dönüştürüyor
outlook=data.iloc[:,0:1].values
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
outlook[:,0]=le.fit_transform(outlook[:,0])

from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(categorical_features='all')
outlook=ohe.fit_transform(outlook).toarray()

windy=data.iloc[:,-2:-1].values
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
windy[:,0]=le.fit_transform(windy[:,0])
windy=windy.astype(np.int64)


play=data.iloc[:,-1:].values
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
play[:,0]=le.fit_transform(play[:,0])

outlook=pd.DataFrame(data=outlook, index=range(14), columns=['o','r','s'])
th=data.iloc[:,1:3].values
th=pd.DataFrame(data=th, index=range(14), columns=['t','h'])
windy=pd.DataFrame(data=windy, index=range(14), columns=['w'])
play=pd.DataFrame(data=play, index=range(14), columns=['p'])


newData=pd.concat([outlook,th],axis=1)
newData=pd.concat([newData,windy],axis=1)
newData=pd.concat([newData,play],axis=1)

#Test ve Train verileri hazırlandı
train=newData.iloc[:,0:4].values
train=pd.DataFrame(data=train, index=range(14), columns=['o','r','s','t'])
newTrain=newData.iloc[:,5:7]
train=pd.concat([train,newTrain],axis=1)

test=newData.iloc[:,4].values
test=pd.DataFrame(data=test, index=range(14), columns=['h',])

#Verileri eğitim ve test için parçaladık
from sklearn.model_selection  import train_test_split
x_train,x_test,y_train,y_test=train_test_split(train,test,test_size=0.33,random_state=0)

#----------------- Multible Linear Regression humidity -----------------
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
#veriyi eğit
regressor.fit(x_train,y_train)
#test et
y_pred=regressor.predict(x_test)



#p value hesaplama ona göre backword yapılacak
import statsmodels.api as sm
X =np.append(arr = np.ones((14,1)).astype(int), values=train ,axis=1 )

X_l = newData.iloc[:,[1,3]].values

model = sm.OLS(test,X_l )
results = model.fit()
print(results.summary())
#ÇIKAN SONUÇ TABLOSUNUN DEĞERLERİ 0,5 ALTINDA OLMALI. P VALUE

