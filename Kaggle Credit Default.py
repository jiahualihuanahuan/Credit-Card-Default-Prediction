#!/usr/bin/env python
# coding: utf-8

# Purpose:
# Predict if a person will default their credit card next month

# In[1]:


import pandas as pd
df=pd.read_csv('D:/Dropbox/Machine Learning/Data/Credit Default/UCI_Credit_Card.csv')
# df=pd.read_csv('/Users/jiahuali1991/Dropbox/Machine Learning/Data/Credit Default/UCI_Credit_Card.csv')
df.sample(5)


# Dataset Information This dataset contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from April 2005 to September 2005.
# 
# Content There are 25 variables:
# 
# ID: ID of each client LIMIT_BAL: Amount of given credit in NT dollars (includes individual and family/supplementary credit SEX: Gender (1=male, 2=female) EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown) MARRIAGE: Marital status (1=married, 2=single, 3=others) AGE: Age in years PAY_0: Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, … 8=payment delay for eight months, 9=payment delay for nine months and above) PAY_2: Repayment status in August, 2005 (scale same as above) PAY_3: Repayment status in July, 2005 (scale same as above) PAY_4: Repayment status in June, 2005 (scale same as above) PAY_5: Repayment status in May, 2005 (scale same as above) PAY_6: Repayment status in April, 2005 (scale same as above) BILL_AMT1: Amount of bill statement in September, 2005 (NT dollar) BILL_AMT2: Amount of bill statement in August, 2005 (NT dollar) BILL_AMT3: Amount of bill statement in July, 2005 (NT dollar) BILL_AMT4: Amount of bill statement in June, 2005 (NT dollar) BILL_AMT5: Amount of bill statement in May, 2005 (NT dollar) BILL_AMT6: Amount of bill statement in April, 2005 (NT dollar) PAY_AMT1: Amount of previous payment in September, 2005 (NT dollar) PAY_AMT2: Amount of previous payment in August, 2005 (NT dollar) PAY_AMT3: Amount of previous payment in July, 2005 (NT dollar) PAY_AMT4: Amount of previous payment in June, 2005 (NT dollar) PAY_AMT5: Amount of previous payment in May, 2005 (NT dollar) PAY_AMT6: Amount of previous payment in April, 2005 (NT dollar) default.payment.next.month: Default payment (1=yes, 0=no) Inspiration Some ideas for exploration:
# 
# How does the probability of default payment vary by categories of different demographic variables? Which variables are the strongest predictors of default payment? Acknowledgements Any publications based on this dataset should acknowledge the following:
# 
# Lichman, M. (2013). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
# 
# The original dataset can be found here at the UCI Machine Learning Repository.

# In[2]:


df.info()


# In[3]:


# check missing value
df.isnull().sum()


# In[4]:


# categorical variables description
df[['SEX','EDUCATION','MARRIAGE']].describe()


# In[5]:


df['SEX'].value_counts(dropna=False)


# In[6]:


df['EDUCATION'].value_counts(dropna=False)


# SEX: Gender (1=male, 2=female) 
# 18112 female and 11888 male
# 
# EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown) 
# University: 14040
# Graduate School: 10585
# High School:4917
# Unknown: 0, 5 and 6 to be combined (280+51+14)
# 
# MARRIAGE: Marital status (1=married, 2=single, 3=others)
# 
# EDUCTAION: undocumented 0 =>solution: change 0 to 5, change 6 to 5
# 
# MARRIAGE: undocumented 0 =>solution: change 0 to 3

# In[7]:


df['SEX'] = df['SEX'].replace(1,'male').replace(2,'female')
df['SEX'].value_counts(dropna=False)


# In[8]:


df['EDUCATION'] = df['EDUCATION'].replace(0,5).replace(6,5).replace(1,'graduate school').replace(2,'university').replace(3,'high school').replace(4,'others').replace(5,'unknown')
df['EDUCATION'].value_counts(dropna=False)


# In[9]:


df['MARRIAGE'] = df['MARRIAGE'].replace(0,3).replace(1,'married').replace(2,'single').replace(3,'others')
df['MARRIAGE'].value_counts(dropna=False)


# In[10]:


df.info()


# In[11]:


# payment delay description
df[['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']].describe()


# PAY_0: Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, … 8=payment delay for eight months, 9=payment delay for nine months and above)
# 
# PAY_?:undocumented -2

# In[12]:


# bill statement description
df[['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']].describe()


# BILL_AMT1: Amount of bill statement in September, 2005 (NT dollar)
# 
# negative values observed: treat it as credit?

# In[13]:


# previous payment description
df[['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']].describe()


# In[14]:


df.LIMIT_BAL.describe()


# LIMIT_BAL: Amount of given credit in NT dollars (includes individual and family/supplementary credit
# 

# In[15]:


# rename two columns
df=df.rename(columns={'default.payment.next.month':'def_pay',
                     'PAY_0':'PAY_1'})
df.columns


# In[16]:


# Calculate default rate
df.def_pay.sum()/len(df.def_pay)


# In[17]:


# prepare X and y for machine learning
y=df['def_pay'].copy()
y.sample(5)


# In[20]:


'''features=['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_1',
       'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']'''
#X = df[features].copy()
X = df.drop(['def_pay','ID'],axis=1)
X.columns


# In[21]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


#categorical_cols = ['SEX', 'EDUCATION', 'MARRIGE']
#numerical_cols = ['LIMIT_BAL', 'AGE', 'PAY_1',
#       'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
#       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
#       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

# we can achive the same goal by 
# determine categorical and numerical features
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object', 'bool']).columns


# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('standardize', StandardScaler())])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


# In[22]:


# split data into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)


# In[23]:


df.def_pay.describe()


# In[24]:


y_train.describe()


# In[25]:


y_test.describe()


# # Dummy Classifier

# In[26]:


from sklearn.dummy import DummyClassifier
clf = DummyClassifier(strategy="most_frequent")
# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', clf)
                             ])

# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

# Preprocessing of test data, get predictions
y_predicted = my_pipeline.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_predicted)


# # Decision Tree Classifier

# In[27]:


import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
param_grid={
    'max_depth':[2,4,5,10,15,20],
    'criterion':['gini','entropy'],
    'max_leaf_nodes':[5,10,20,50,100],
    'min_samples_split':[5,10,15,20]
}

grid_tree = GridSearchCV(DecisionTreeClassifier(),param_grid,cv=5,scoring='accuracy',n_jobs=-1)



# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('grid_search', grid_tree)
                             ])

# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

print(grid_tree.best_estimator_)
print(np.abs(grid_tree.best_score_))


# # K Nearest Neighbour

# In[28]:


import numpy as np
from sklearn.model_selection import GridSearchCV
param_grid={
    'n_neighbors':[1,3,5,10],
    'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
    'weights':['uniform', 'distance'],
    'leaf_size':[2,10,20]
}

from sklearn.neighbors import KNeighborsClassifier
grid = GridSearchCV(KNeighborsClassifier(),param_grid,cv=3,scoring='accuracy',verbose=1,n_jobs=-1)

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('grid_search', grid)
                             ])

# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

print(grid.best_estimator_)
print(np.abs(grid.best_score_))


# # Support Vector Machine

# In[35]:


import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
param_grid={
    'C':[0.1,1,10,100],
    'gamma':[0.1,0.01,0.001,0.0001],
    'kernel':['rbf']
}

grid = GridSearchCV(SVC(),param_grid,cv=3,scoring='accuracy',verbose=1,n_jobs=-1)

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('grid_search', grid)
                             ])

# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

print(grid.best_estimator_)
print(np.abs(grid.best_score_))


# In[36]:


from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
param_grid={
    'C':[0.1,1,10],
    'tol':[0.01,0.001,0.0001]
}

grid = GridSearchCV(LinearSVC(),param_grid,cv=3,scoring='accuracy',verbose=1,n_jobs=-1)

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('grid_search', grid)
                             ])

# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

print(grid.best_estimator_)
print(np.abs(grid.best_score_))


# # logistic Regression

# In[60]:


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
param_grid={
    'solver':['lbfgs', 'liblinear', 'sag', 'saga'],
    'C':[0.01,0.1,1,10,100],
    'tol':[0.000001,0.0001,0.01,1,100]
}

grid_log = GridSearchCV(LogisticRegression(),param_grid,cv=3,scoring='accuracy',verbose=1, n_jobs=-1)

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('grid_search', grid_log)
                             ])


# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

# displace best hyperparameters and best accuracy
print(grid_log.best_estimator_)
print(np.abs(grid_log.best_score_))


# # Random Forest

# In[62]:


from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import GridSearchCV
param_grid={
    'criterion':['gini', 'entropy'],
    'max_depth':[1,10,100,1000]
}

grid_rf = GridSearchCV(RandomForestClassifier(),param_grid,cv=3,scoring='accuracy',verbose=1, n_jobs=-1)

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('grid_search', grid_rf)
                             ])


# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

# displace best hyperparameters and best accuracy
print(grid_rf.best_estimator_)
print(np.abs(grid_rf.best_score_))


# # XGBoost

# In[41]:


# XGBoost
from xgboost import XGBClassifier
import numpy as np
from sklearn.model_selection import GridSearchCV
param_grid={
    'booster':['gbtree', 'gblinear', 'dart'],
    'max_depth':[1,10,100,1000]
}

grid = GridSearchCV(XGBClassifier(),param_grid,cv=3,scoring='accuracy',verbose=1, n_jobs=-1)

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('grid_search', grid)
                             ])


# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

# displace best hyperparameters and best accuracy
print(grid.best_estimator_)
print(np.abs(grid.best_score_))


# In[44]:


# XGBoost
from xgboost import XGBClassifier
import numpy as np
from sklearn.model_selection import GridSearchCV

clf = XGBClassifier()

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', clf)
                             ])


# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

# Preprocessing of test data, get predictions
y_predicted = my_pipeline.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_predicted)


# # Naive Bayes

# In[46]:


from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
import numpy as np
from sklearn.model_selection import GridSearchCV

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', clf)
                             ])


# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

# Preprocessing of test data, get predictions
y_predicted = my_pipeline.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_predicted)


# # Ridge Classifier

# In[47]:


from sklearn.linear_model import RidgeClassifier
clf = RidgeClassifier()
import numpy as np
from sklearn.model_selection import GridSearchCV

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', clf)
                             ])


# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

# Preprocessing of test data, get predictions
y_predicted = my_pipeline.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_predicted)


# # Stochastic Gradient Descent Classifier

# In[49]:


from sklearn.linear_model import SGDClassifier
clf = SGDClassifier()
import numpy as np
from sklearn.model_selection import GridSearchCV

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', clf)
                             ])


# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

# Preprocessing of test data, get predictions
y_predicted = my_pipeline.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_predicted)


# # Neural Network

# In[50]:


from sklearn.neural_network import MLPClassifier
clf = MLPClassifier()
import numpy as np
from sklearn.model_selection import GridSearchCV

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', clf)
                             ])


# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

# Preprocessing of test data, get predictions
y_predicted = my_pipeline.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_predicted)


# In[ ]:




