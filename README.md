# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary python packages using import statements.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Split the dataset using train_test_split.

4.Calculate Y_Pred and accuracy.

5.Print all the outputs.

6.End the Program.

## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: PULI NAGA NEERAJ
RegisterNumber:  212223240130
```
```
import chardet
file='spam.csv'
with open (file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
# Output:

![image](https://github.com/PuliNagaNeeraj/Implementation-of-SVM-For-Spam-Mail-Detection/assets/138849173/64f65c66-8dc8-44ac-8c69-e2f09c306d9c)
### Head(}
![image](https://github.com/PuliNagaNeeraj/Implementation-of-SVM-For-Spam-Mail-Detection/assets/138849173/c031c623-b19b-47ed-b088-d46448ff445b)
### info()
![image](https://github.com/PuliNagaNeeraj/Implementation-of-SVM-For-Spam-Mail-Detection/assets/138849173/57e1ef81-9b63-4c9f-a804-772a00e97dbc)
### isnull.sum
![image](https://github.com/PuliNagaNeeraj/Implementation-of-SVM-For-Spam-Mail-Detection/assets/138849173/d0ebe9dd-2192-401a-9ddf-6e3117adf7cd)
### prediction of y
![image](https://github.com/PuliNagaNeeraj/Implementation-of-SVM-For-Spam-Mail-Detection/assets/138849173/94808240-4e9e-4cc1-95ac-65d1c79c5ac9)
### Accuracy
![image](https://github.com/PuliNagaNeeraj/Implementation-of-SVM-For-Spam-Mail-Detection/assets/138849173/ab495f5f-6080-4240-b311-55a88368361c)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
