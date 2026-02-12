# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import the required libraries.
2. Upload and read the dataset
3. check for any null values using the isnull() function.
4. Form sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn

## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: YAAZHINI S
RegisterNumber:  212224230308


import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()

data.info()

data.isnull().sum()

data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['salary']=le.fit_transform(data['salary'])
data.head()

x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()

y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy

dt.predict([[0.5,0.8,9,206,6,0,1,2]])
```

## Output:

## data head

<img width="1363" height="227" alt="exp8(1)" src="https://github.com/user-attachments/assets/42276d76-c98c-41f9-906c-c7e03c771a18" />

## Data set info

<img width="2230" height="590" alt="exp8(2)" src="https://github.com/user-attachments/assets/7e345c07-ef10-4a30-a95f-b0cf1056978a" />

## Null Set

<img width="2230" height="406" alt="exp8(3)" src="https://github.com/user-attachments/assets/2a77e882-ab14-4240-8a1f-d489f083db97" />

## Values count

<img width="1112" height="82" alt="exp8(4)" src="https://github.com/user-attachments/assets/e21af417-aa0d-4e61-8b5f-b74645637e8e" />

## Dataset transformed head

<img width="1118" height="202" alt="exp8(5)" src="https://github.com/user-attachments/assets/bc1dddb5-5923-4bf5-a688-3ede72d0e9c9" />

## X.Head

<img width="1111" height="177" alt="exp8(6)" src="https://github.com/user-attachments/assets/669b431e-4758-4cc9-a467-67dc558a2297" />

## Accuracy

<img width="1090" height="32" alt="exp8(7)" src="https://github.com/user-attachments/assets/f4681073-cc3f-49bc-94f0-50fa466bfd4c" />

## data prediction

<img width="872" height="641" alt="exp8(8)" src="https://github.com/user-attachments/assets/06600f12-3fc6-4d0b-ac8e-5741eee2c387" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
