#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 01:50:22 2020

@author: abhishek
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.metrics import r2_score
import pickle

def training():
    data = pd.read_csv("My.csv")
    data.head(10)
    len(data.Course)
    pd.unique(data.Course)
    
    data = data.drop(["0"],axis=1)
    
    data.groupby('Course')['Termid'].count()
    
    imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
    imputer.fit(data.iloc[:,4:8].values)
    data.iloc[:,4:8] = imputer.transform(data.iloc[:,4:8].values)
    
    map1={'O':10,'A+':9,'A':8,'B+':7,'B':6,'C':5,'D':4,'E':3,'F':2}
    data['Grade']=data['Grade'].map(map1)
    print(data['Grade'])
    
    from sklearn.preprocessing import LabelEncoder,OneHotEncoder
    labelencoder=LabelEncoder()
    
    data['Gender'] = labelencoder.fit_transform(data['Gender'])
    data['Course'] = labelencoder.fit_transform(data['Course'])
    
    X = data.iloc[:,2:8]
    X = X.drop(['Grade'],axis=1)
    
    y = data.iloc[:,[3]]
    
    
    from sklearn.model_selection import train_test_split
    
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
    
    from sklearn.svm import SVC
    
    
    svm=SVC(kernel='linear')
    svm.fit(x_train,y_train)
    x_s=svm.predict(x_test)
    print(r2_score(x_s,y_test)*100)
    
    from sklearn.metrics import confusion_matrix
    cm=confusion_matrix(y_test,x_s)
    print(cm)
    
    u_input = np.array([[1,7, 5, 5, 32]])
    
    pred = svm.predict(u_input)
    pickle.dump(svm,open('Filepickle.sav','wb'))


training()
  
    
