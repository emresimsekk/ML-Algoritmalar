# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 18:55:08 2020

@author: Emre
"""

import numpy as np
import pandas as pd

#Verilerin Yüklenmesi
dataset = pd.read_csv('rea.csv', error_bad_lines=False)

import re
import nltk
# Stemmer (Kelimeleri gövdeleri haline getirme)
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

# stopWords (anlamsız kelimeleri çıkarma)
nltk.download('stopwords')
from nltk.corpus import stopwords


digest=[]
for i in range(716):
    # Regular Expression (Yorumlardaki noktalama işaretlerini değiştirme)
    comment=re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    # Lower (Küçük harfe çevirme)
    comment=comment.lower()
    # Split (Kelimeleri listeye çevirme)
    comment=comment.split()
    # Stopword al kümeye çevir kümelerinde içinde kelime yoksa gövdesini bul 
    comment=[ps.stem(word) for word in comment if not word in set(stopwords.words('english'))]
    comment=' '.join(comment)
    digest.append(comment)
    
# Kelimeleri sayısını sayıyor
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=2000)

# Bağımsız Değişken
X=cv.fit_transform(digest).toarray()
# Bağımlı Değişken
Y=dataset.iloc[:,1].values

#Verileri eğitim ve test için parçaladık
from sklearn.model_selection  import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.33,random_state=0)

# Naive Bayes 
from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()

gnb.fit(X_train,Y_train)

y_pred=gnb.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,y_pred)
print(cm)












