#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


# In[ ]:


data = pd.read_csv('adult.data', header=None)
data.set_axis(['age','workclass','fnlwgt','education','ed_num','marital-status',
               'occupation','relationship','race','sex','capital-gain','capital-loss',
                'hours_per_week','native-country','salary'], axis=1, inplace=True)

X = data.drop(['fnlwgt', 'ed_num', 'capital-gain', 'capital-loss', 'salary'], axis=1)
y = data['salary']

X


# The following custom class encodes all the dataframe as suitable if ordinal and not ordinal features are previously defined. You can also decide to keep or not NaNs.

# In[ ]:


from custom_pipeline_classes import Whole_MultiEncoder, custom_scale

not_ordinal_cat = ['sex','race','workclass','education','native-country',
                  'occupation','marital-status','relationship']
ordinal_cat = None
continous =['age','hours_per_week']

# This class encodes the all dataframe based on the columns passed as argument 
my_enc = Whole_MultiEncoder(non_ordinal = not_ordinal_cat,
                         ordinal = ordinal_cat, keep_nan=False)

y =  LabelEncoder().fit_transform(y)
y = pd.Series(y, name='target')

scaler = custom_scale(continous)  #scales only some columns of the df 


# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)


# In[ ]:


pipe = Pipeline([('encoder', my_enc),
                 ('scaler',scaler),
                 ('model', RandomForestClassifier())])

pipe.fit(X_train, y=y_train)

pipe.score(X_test, y_test)


# In[ ]:


pipe_1 = Pipeline([('encoder', my_enc),
                 ('scaler',scaler),
                 ('model', DecisionTreeClassifier())])

pipe_1.fit(X_train, y_train)
pipe_1.score(X_test, y_test)


# In[ ]:


# Different approach
from sklearn.pipeline import make_pipeline
pipe_2 = make_pipeline(my_enc, scaler , SVC(gamma='auto'))

pipe_2.fit(X_train, y_train)
pipe_2.score(X_test, y_test)


# In[ ]:




