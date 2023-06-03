# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages.
2. Analyse the data.
3. Use label encoder for position.
4. Apply decision tree regressor for and find MSE values .
5.Display the results.
## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: J.Archana priya
RegisterNumber:  212221230007
*/
```
```
import pandas as pd
data = pd.read_csv(("/content/Salary.csv"))
data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder()

data["Position"] = le.fit_transform(data["Position"])
data.head()

x = data[["Position","Level"]]
y  = data[["Salary"]]
x.head()


y.head()

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain ,ytest = train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(xtrain,ytrain)
ypred = dt.predict(xtest)

from sklearn import metrics
mse = metrics.mean_squared_error(ytest,ypred)
mse

r2 = metrics.r2_score(ytest,ypred)
r2

dt.predict([[5,6]])

```

## Output:
### data.head()
![image](https://github.com/Archana2003-Jkumar/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/93427594/b6e42c37-f806-4fff-b0bc-7f503b5ac0eb)
### data.info()
![image](https://github.com/Archana2003-Jkumar/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/93427594/c6bb1b0a-2152-4753-913a-c01b65592bb6)
### isnull() and sum()
![image](https://github.com/Archana2003-Jkumar/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/93427594/d1e55d92-8834-4d5c-94b9-b0000fdaf927)
### data.head() for salary
![image](https://github.com/Archana2003-Jkumar/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/93427594/f197095e-729d-4244-95ae-7a15e92fff92)
### MSE value 
![image](https://github.com/Archana2003-Jkumar/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/93427594/f5656c7e-dec6-4dd5-b9bd-51058d91f648)
### r2 value
![image](https://github.com/Archana2003-Jkumar/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/93427594/b045b5ad-0a90-414c-9c33-fcc5a182c233)
### data prediction
![image](https://github.com/Archana2003-Jkumar/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/93427594/2888d863-ac83-45a0-8853-b22b6c6578b6)
## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
