#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as p
import numpy as np
import matplotlib.pyplot as plt
df=p.read_csv('student_scores.csv')
X = df.iloc[:, :1].values
y = df.iloc[:, :1].values
print(df)
print(df.head())
print(df.columns)
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
viz_train = plt
viz_train.scatter(X_train, y_train, color='red')
viz_train.plot(X_train, regressor.predict(X_train), color='blue')
viz_train.title('Hours VS Score (Training set)')
viz_train.xlabel('Hours')
viz_train.ylabel('Score')
viz_train.show()
viz_test = plt
viz_test.scatter(X_test, y_test, color='red')
viz_test.plot(X_train, regressor.predict(X_train), color='blue')
viz_test.title('Hours VS Score (Test set)')
viz_test.xlabel('Hours')
viz_test.ylabel('Score')
viz_test.show()
y_pred = regressor.predict(X_test)
print(y_pred)


# In[ ]:




