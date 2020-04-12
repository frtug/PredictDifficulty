#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 16:33:55 2020

@author: abhishek
"""
import pickle
import numpy as np
from sklearn.svm import SVC
 

def prediction(inpu_var):
    classi = pickle.load(open('Filepickle.sav','rb'))
    ped = classi.predict(inpu_var)
    
    final_result = ped[0]/10
    print(final_result)
    if (final_result > 0.9):
        print("Course is Easy to score")
    elif(final_result > 0.75):
        print("Course Difficuly Intermediate level")
    elif (final_result > 0.5):
        print("Course is Hard")
    else :
        print("Course is Very hard")
        
    
inpu_var = list(map(int, input("Enter a multiple value: ").split())) 
inpu_var = np.array([inpu_var])
prediction(inpu_var)
