#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[2]:


data = pd.read_csv('heart.csv')


# In[3]:


data


# #  Outliers Detection and Removal

# In[4]:


data_cont = data[['age','trestbps','chol','thalach','oldpeak']]
for figure in data_cont:
    plt.figure()
    plt.title(figure)
    ax = sns.boxplot(data[figure])


# In[5]:


from sklearn.preprocessing import StandardScaler


# In[6]:


scaler_1 = StandardScaler(with_mean=True, with_std=True)  # 0 mean for z_score
scaler_2 = StandardScaler(with_mean=False, with_std=True)  # actual mean 


# In[7]:


no_mean_data = data.copy()


# In[8]:


no_mean_data[['age','trestbps','chol','thalach','oldpeak']] = scaler_1.fit_transform(data[['age','trestbps','chol','thalach','oldpeak']])


# In[9]:


from scipy import stats


# In[10]:


z= np.abs(stats.zscore(no_mean_data))


# In[11]:


row_to_remove = np.where(z > 3) # with z <= 2 there are no outliers


# In[12]:


data = data.drop(row_to_remove[0], axis=0)


# In[13]:


data[['age','trestbps','chol','thalach','oldpeak']] = scaler_2.fit_transform(data[['age','trestbps','chol','thalach','oldpeak']])


# # Feature Selection
# ## We will try different methods 
# ### Univariate feature selection

# In[14]:


from sklearn.feature_selection import chi2, f_classif
from math import log10


# In[15]:


cat = data[[data.columns[i] for i in range(len(data.columns)) if data.iloc[:,i].dtypes == 'int64']] 
cont = data[[data.columns[i] for i in range(len(data.columns)) if data.iloc[:,i].dtypes == 'float64']]


# In[16]:


chi_scores, chi_pval = chi2(cat.drop('target', axis=1), cat['target'])
log10_pval = -np.log10(chi_pval)
columns = cat.drop('target', axis=1).columns
pd.DataFrame(log10_pval, index=columns).plot.bar()


# In[17]:


f_scores, f_pval = f_classif(cont, cat['target'])
log10_f_pval = -np.log10(f_pval)
pd.DataFrame(log10_f_pval, index=cont.columns).plot.bar()


# fbs probably is not a good predictor

# In[18]:


datasets = []


# In[19]:


univariate_dataset = data.drop(['fbs','target'], axis=1)
univariate_dataset.name = 'univariate_dataset'
datasets.append(univariate_dataset)


# ### SelectFromModel

# In[20]:


from sklearn.feature_selection import SelectFromModel


# In[21]:


X = data.iloc[:,0:-1]
y = data.iloc[:,-1]


# In[22]:


from sklearn.svm import LinearSVC

lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
model = SelectFromModel(lsvc, prefit=True)
SelectFM_SVC_dataset = model.transform(X)
SelectFM_SVC_dataset = pd.DataFrame(SelectFM_SVC_dataset)
SelectFM_SVC_dataset.name = 'SelectFM_SVC_dataset'

datasets.append(SelectFM_SVC_dataset)


# In[23]:


from sklearn.ensemble import ExtraTreesClassifier

tree_clf = ExtraTreesClassifier(n_estimators=70)
tree_clf = tree_clf.fit(X, y)
model = SelectFromModel(tree_clf, prefit=True)
SelectFM_Tree_dataset = model.transform(X)
SelectFM_Tree_dataset = pd.DataFrame(SelectFM_Tree_dataset)
SelectFM_Tree_dataset.name = 'SelectFM_Tree_dataset'


datasets.append(SelectFM_Tree_dataset)


# ### Sequential Feature Selection

# In[24]:


from sklearn.feature_selection import SequentialFeatureSelector


# In[25]:


lsvc = LinearSVC(C=0.01, penalty="l1", dual=False)
tree_clf = ExtraTreesClassifier(n_estimators=70)


# In[26]:


sfs = SequentialFeatureSelector(lsvc, n_features_to_select=9)
sfs_back = SequentialFeatureSelector(lsvc, n_features_to_select=9, direction='backward')

Sequential_SVC_dataset = sfs.fit_transform(X, y)
Sequential_SVC_dataset = pd.DataFrame(Sequential_SVC_dataset)
Sequential_SVC_dataset.name = 'Sequential_SVC_dataset'


datasets.append(Sequential_SVC_dataset)

Sequential_back_SVC_dataset = sfs_back.fit_transform(X, y) 
Sequential_back_SVC_dataset = pd.DataFrame(Sequential_back_SVC_dataset)
Sequential_back_SVC_dataset.name = 'Sequential_back_SVC_dataset'


datasets.append(Sequential_back_SVC_dataset)


# In[27]:


sfs = SequentialFeatureSelector(tree_clf, n_features_to_select=9)
sfs_back = SequentialFeatureSelector(tree_clf, n_features_to_select=9, direction='backward')

Sequential_Tree_dataset = sfs.fit_transform(X, y)
Sequential_Tree_dataset = pd.DataFrame(Sequential_Tree_dataset)
Sequential_Tree_dataset.name = 'Sequential_Tree_dataset'


datasets.append(Sequential_Tree_dataset)

Sequential_back_Tree_dataset = sfs_back.fit_transform(X, y) 
Sequential_back_Tree_dataset = pd.DataFrame(Sequential_back_Tree_dataset)
Sequential_back_Tree_dataset.name = 'Sequential_back_Tree_dataset'


datasets.append(Sequential_back_Tree_dataset)


# In[28]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
import xgboost as xgb


# In[29]:


models = [
    LogisticRegression(random_state=0, max_iter=1000),
    DecisionTreeClassifier(random_state=0, max_depth=5),
    RandomForestClassifier(max_depth=5, random_state=0),
    SVC(gamma='auto', kernel='linear'),
    xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False,
                              n_estimators = 10000, n_jobs=10, early_stopping_rounds=10),
    
]


# In[30]:


models_names = [
    'Logistic Regression',
    'Decision Tree',
    'Random Forest',
    'SVC',
    'XGBoost'
]


# In[31]:


results = {}
for dtst in datasets:
    scores = {}
    counter = 0
    for model in models:
        score = cross_val_score(model, dtst, y, cv=3)
        scores[models_names[counter]] = np.mean(score)
        counter += 1
    results[dtst.name] = scores   


# In[32]:


results_accuracy = pd.DataFrame(results)


# In[33]:


results = {}
for dtst in datasets:
    scores = {}
    counter = 0
    for model in models:
        score = cross_val_score(model, dtst, y, cv=3, scoring='roc_auc')
        scores[models_names[counter]] = np.mean(score)
        counter += 1
    results[dtst.name] = scores   


# In[34]:


results_ROC_AUC = pd.DataFrame(results)


# In[35]:


results_accuracy


# In[37]:


results_ROC_AUC


# In[36]:


model_accuracy = pd.DataFrame(results_accuracy.mean(axis=1)).rename(columns={0:'mean_accuracy'})
dataset_accuracy = pd.DataFrame(results_accuracy.mean(axis=0)).rename(columns={0:'mean_accuracy'})
model_ROC_AUC = pd.DataFrame(results_ROC_AUC.mean(axis=1)).rename(columns={0:'mean_roc_auc_score'})
dataset_ROC_AUC = pd.DataFrame(results_ROC_AUC.mean(axis=0)).rename(columns={0:'mean_roc_auc_score'})


# In[39]:


model_scoring = model_accuracy.join(model_ROC_AUC)
dataset_scoring = dataset_accuracy.join(dataset_ROC_AUC)


# In[40]:


model_scoring


# In[41]:


dataset_scoring


# In[ ]:




