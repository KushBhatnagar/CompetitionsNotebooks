# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 08:24:44 2019

@author: Kush
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv('train.csv')
X=dataset[['OverallQual','YearBuilt','TotalBsmtSF','GrLivArea','GarageCars']]
y=dataset[['SalePrice']]


# Fitting Multiple linear regression to the data set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X,y)
y_pred=regressor.predict(X)


#Get test data 
dataset_test=pd.read_csv('test.csv')
X_test=dataset_test[['OverallQual','YearBuilt','GrLivArea','TotalBsmtSF','GarageCars']]
y_test_id=dataset_test[['Id']]
y_pred=regressor.predict(X)

#Checking missing value in test data set
total_missing_values_X_test=X_test.isnull().sum().sort_values(ascending=False)
total_missing_values_X_test

#Checking the missing Garage Cars record
X_test[X_test['GarageCars'].isnull()]
#Checking the missing Total Bsmt SF record
X_test[X_test['TotalBsmtSF'].isnull()]

X_test.at[1116,'GarageCars'] = 2
X_test.at[660,'TotalBsmtSF'] = 1046.12

X_test.sort_values(by='GrLivArea',ascending=False)[:2]

#Drop the first index from top two
X_test=X_test.drop(1089)
y_test_pred=regressor.predict(X_test)

#Converting predicted values into dataframe
y_pred_df=pd.DataFrame(y_pred,columns=['SalePrice'])
y_test_pred_df=pd.DataFrame(y_test_pred,columns=['SalePrice'])

#Creating comparision plot between Actual and predicted value with test data set
plt.figure(figsize=(10,10))
sns.lineplot(data=y,palette ='rainbow')
sns.lineplot(data=y_test_pred_df)

#Concating id and predicted data
submission=pd.concat([y_test_id,y_test_pred_df],ignore_index=True)