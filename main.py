from flask import Flask,redirect,url_for,render_template,request
import os
import pickle
import numpy as np
from sklearn.svm import SVC
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


app = Flask(__name__)

@app.route('/')
def home():
     return render_template('index.html',content=['Abhishek','shalu','Shubham'])

@app.route("/predictt", methods = ['GET','POST'])
def predictt():
     course = int(request.form.get('Course'))
     ca_marks = int(request.form.get('caname'))
     mt_marks = int(request.form.get('mtname'))
     etp_marks = int(request.form.get('etpname'))
     ett_marks = int(request.form.get('ettname'))
     sel = [course,ca_marks,mt_marks,etp_marks,ett_marks]
     inpu_var = np.array([sel])
     classi = pickle.load(open('Filepickle.sav','rb'))
     ped = classi.predict(inpu_var)
     
     final_result = ped[0]/10
     print(final_result)
     if (final_result > 0.9):
          return "Course is Easy to score"
     elif(final_result > 0.75):
          return "Course Difficuly Intermediate level"
     elif (final_result > 0.5):
          return "Course is Hard"
     else :
          return "Course is Very hard"
     return "Failed to Predict"

@app.route("/training",methods = ['GET','POST'])
def training():
     data = pd.read_csv("My.csv")
     data = data.drop(["0"],axis=1)
     imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
     imputer.fit(data.iloc[:,4:8].values)
     data.iloc[:,4:8] = imputer.transform(data.iloc[:,4:8].values)
     map1={'O':10,'A+':9,'A':8,'B+':7,'B':6,'C':5,'D':4,'E':3,'F':2}
     data['Grade']=data['Grade'].map(map1)
     labelencoder=LabelEncoder()
     data['Gender'] = labelencoder.fit_transform(data['Gender'])
     data['Course'] = labelencoder.fit_transform(data['Course'])

     X = data.iloc[:,2:8]
     X = X.drop(['Grade'],axis=1)
     y = data.iloc[:,[3]]

     x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
     svm=SVC(kernel='linear')
     svm.fit(x_train,y_train)
     x_s=svm.predict(x_test)
     pickle.dump(svm,open('Filepickle.sav','wb'))

     return "Accuracy is "+str(r2_score(x_s,y_test)*100)



if __name__ == "__main__":
     app.run(debug=True)