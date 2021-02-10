#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, precision_score, recall_score


# In[2]:


data = pd.read_csv('weatherAUS.csv')


# In[3]:


data


# In[4]:


data[data.columns]


# In[5]:


data.info()


# In[6]:


data['RainTomorrow'].value_counts()


# In[7]:


data['RainTomorrow'].isna().value_counts()


# Dataset is imbalanced and there are several missing values both in the X features and between the target values. We will evaluate the model in the following different dataset configurations:
# - Without the examples relative to the target missing values, with the X missing values handled (imputed) by xgboost;
# - With all the missing values imputed by kNN Imputer from sklearn.
# 
# We will also handle the imbalance by:
# - Stratifying;
# - Ensembling the resampled dataset.

# In[8]:


missing = data.isna().sum().sort_values()
missing_ratio = data.isna().sum()/data.isna().count()
missing_matrix = pd.concat([missing, missing_ratio], axis=1)


# In[9]:


#Drop the features with more than one third of data missing

drop_indexes = missing_matrix[missing_matrix[1] > 0.33].index


# In[10]:


processed_data = data.drop(drop_indexes, axis=1)


# In[11]:


#Select only data with target values not missing
X_y = processed_data[processed_data['RainTomorrow'].isna() == False]


# Testing with chi squared and anova to see if some features can be discarded

# In[12]:


cat = X_y[['Location','WindGustDir','WindDir9am','WindDir3pm','RainToday','RainTomorrow']]


# In[13]:


from sklearn.feature_selection import chi2, f_classif


# In[14]:


cat.dropna(inplace=True)


# In[15]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
cat = cat.apply(LabelEncoder().fit_transform)


# In[16]:


chi_scores = chi2(cat.drop(['RainTomorrow'], axis=1), cat['RainTomorrow'])


# In[17]:


cont = data.drop(['Location','WindGustDir','WindDir9am','WindDir3pm','RainToday','RainTomorrow'], axis=1)
cont['RainTomorrow'] = cat['RainTomorrow']


# In[18]:


cont.dropna(inplace=True)
cont.drop('Date', axis=1, inplace=True)


# In[19]:


anova_scores = f_classif(cont.drop(['RainTomorrow'], axis=1), cont['RainTomorrow'])


# In[20]:


p_values = pd.Series(chi_scores[1],index = cat.columns[0:-1])
p_values.plot.bar();


# In[21]:


f_p_values = pd.Series(anova_scores[1],index = cont.columns[0:-1])
f_p_values.plot.bar();


# The test's p-values indicate that location must be discarded since its presence it's not statistically significant if we choose alpha = 0.05. Despite of this, location can be important since in certain areas some conditions could give rain while in other not. Furthermore alpha can be choosen higher since Type I errors do not lead to dramatic effects in this case (e.g. counting location when it's not needed). For the sake of curiosity, for once, we will produce also the results relative to not taking location into account. 

# In[22]:


# Encoding and leaving NaN values

from Custom_Class_Func import MultiColumnLabelEncoder


# In[23]:


# we will mask to keep NaN's and replace after encoding
original = X_y
mask = X_y.isnull()


# In[24]:


X_y = MultiColumnLabelEncoder(columns=['Location','WindGustDir','WindDir9am',
                                'WindDir3pm','RainToday','RainTomorrow']).fit_transform(X_y.astype(str))


# In[25]:


X_y = X_y.where(~mask, original).drop(['Date'],axis=1)


# In[26]:


X_y.isna().values.any()  #checking if masking was successful


# In[27]:


X_y = X_y.astype(float)
X = X_y.drop('RainTomorrow', axis=1)
y = X_y['RainTomorrow']


# # No Imputing

# ## Stratifying

# In[28]:


from sklearn.model_selection import train_test_split


# In[29]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, stratify=y )


# In[30]:


import xgboost as xgb


# In[31]:


# we will use n_jobs to parallelize computation among different threads 
# (in my pc max 12, but empiric evidence shows best with 10)

model = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, n_estimators = 10000, n_jobs=10)
model.fit(X_train, y_train, verbose=True,
            early_stopping_rounds=10, eval_metric='auc', eval_set=[(X_test, y_test)])


# In[32]:


y_pred = model.predict(X_test)
results = pd.Series({'AUC_ROC' : roc_auc_score(y_test, y_pred), 'Precision' : precision_score(y_test, y_pred),
        'Recall' : recall_score(y_test, y_pred)}, name='No_imput_Stratify')
                                
print(confusion_matrix(y_test, y_pred)/y_pred.shape, '\n')
print('Area under ROC curve:', results['AUC_ROC'])
print('Precision:',results['Precision'])
print('Recall:', results['Recall'])


# ### Fine Tuning

# In[33]:


from sklearn.model_selection import GridSearchCV 


# In[34]:


#fine tuning, taking imbalance in account
param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.1, 0.01, 0.05],
    'gamma': [0, 0.25, 1.0],
    'reg_lambda': [0, 1.0, 10.0],
    'scale_pos_weight': [3, 5, 44] # as XGBoost recommends: sum(negative instances) / sum(positive instances)
}


# In[35]:


optimal_params = GridSearchCV(
    estimator=xgb.XGBClassifier(objective='binary:logistic', n_estimators = 10000,
                               subsample=0.9, colsample_bytree=0.5),
    param_grid=param_grid,
    scoring='roc_auc',
    verbose=False,
    n_jobs = 10,
    cv = 3)


# In[36]:


optimal_params.fit(X_train, 
                   y_train, 
                   early_stopping_rounds=10,                
                   eval_metric='auc',
                   eval_set=[(X_test, y_test)],
                   verbose=False)
print(optimal_params.best_params_)


# In[40]:


# run again tuning with borderline params from before
param_grid = {
    'max_depth': [5],
    'learning_rate': [0.05],
    'gamma': [0.25],
    'reg_lambda': [10.0],
    'scale_pos_weight': [2,3] 
}


# In[41]:


optimal_params = GridSearchCV(
    estimator=xgb.XGBClassifier(objective='binary:logistic', n_estimators = 10000,
                               subsample=0.9, colsample_bytree=0.5),
    param_grid=param_grid,
    scoring='roc_auc',
    verbose=False,
    n_jobs = 10,
    cv = 3)


# In[42]:


optimal_params.fit(X_train, 
                   y_train, 
                   early_stopping_rounds=10,                
                   eval_metric='auc',
                   eval_set=[(X_test, y_test)],
                   verbose=False)
print(optimal_params.best_params_)


# In[43]:


# The previous params are confirmed
model = xgb.XGBClassifier(objective='binary:logistic', gamma=0.25, learn_rate=0.05,
                        max_depth=5, reg_lambda=10, scale_pos_weight=2, n_estimators=10000, n_jobs=10)
model.fit(X_train, y_train, verbose=True,
            early_stopping_rounds=10, eval_metric='auc', eval_set=[(X_test, y_test)])


# In[44]:


y_pred = model.predict(X_test)

a = {'AUC_ROC' : roc_auc_score(y_test, y_pred), 'Precision' : precision_score(y_test, y_pred),
        'Recall' : recall_score(y_test, y_pred)}

results = pd.concat([results, pd.Series(a, name='No_imput_Stratify_optimized')], axis=1)


# In[45]:


print(confusion_matrix(y_test, y_pred)/y_pred.shape, '\n')
print('Area under Roc curve:', a['AUC_ROC'])
print('Precision:',a['Precision'])
print('Recall:', a['Recall'])


# ## Ensembled resampling
# We will use **custom functions** to compute predictions.

# In[46]:


from Custom_Class_Func import get_resampled_datasets, get_xgboost_ensemble_predictions, ensemble_xgboost_CV


# In[47]:


a = ensemble_xgboost_CV(X_y, threads=10)


# In[48]:


results = pd.concat([results, pd.Series(a, name='No_imput_Ensembled')], axis=1)


# # kNN Imputer
# ## Stratifying

# In[49]:


# pick the dataset with missing values in target values
X_y = processed_data


# In[50]:


original = X_y
mask = X_y.isnull()
X_y = MultiColumnLabelEncoder(columns=['Location','WindGustDir','WindDir9am',
                                'WindDir3pm','RainToday','RainTomorrow']).fit_transform(X_y.astype(str))
X_y = X_y.where(~mask, original).drop(['Date'],axis=1)
X_y = X_y.astype(float)


# In[53]:


from sklearn.impute import KNNImputer


# In[54]:


imputer = KNNImputer(n_neighbors=4)
imputed_dataset = imputer.fit_transform(X_y)


# In[55]:


columns = X_y.columns


# In[56]:


X_y = pd.DataFrame(imputed_dataset, columns=columns)


# In[57]:


X = X_y.drop('RainTomorrow', axis=1)
y = X_y['RainTomorrow']

# Because imputer sometimes gives values between 1 and 0 and the majority of y values are 0:
y = y.apply(lambda x: 0 if x <= 0.5 else 1)  


# In[58]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, stratify=y )


# In[59]:


model_1 = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, n_estimators = 10000, n_jobs=10)
model_1.fit(X_train, y_train, verbose=True,
            early_stopping_rounds=10, eval_metric='auc', eval_set=[(X_test, y_test)])


# In[60]:


y_pred = model_1.predict(X_test)
a = {'AUC_ROC' : roc_auc_score(y_test, y_pred), 'Precision' : precision_score(y_test, y_pred),
        'Recall' : recall_score(y_test, y_pred)}

results = pd.concat([results, pd.Series(a, name='kNNImput_Stratify')], axis=1)


# In[61]:


print(confusion_matrix(y_test, y_pred)/y_pred.shape)
print('Area under Roc curve:', roc_auc_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))


# ### Fine Tuning

# In[62]:


param_grid = {
    'max_depth': [5],
    'learning_rate': [0.01, 0.05],
    'gamma': [0, 0.25, 1.0],
    'reg_lambda': [1.0, 10.0],
    'scale_pos_weight': [2, 3] # as XGBoost recommends: sum(negative instances) / sum(positive instances)
}


# In[63]:


optimal_params = GridSearchCV(
    estimator=xgb.XGBClassifier(objective='binary:logistic', n_estimators = 10000,
                               subsample=0.9, colsample_bytree=0.5),
    param_grid=param_grid,
    scoring='roc_auc',
    verbose=False,
    n_jobs = 10,
    cv = 3)


# In[64]:


optimal_params.fit(X_train, 
                   y_train, 
                   early_stopping_rounds=10,                
                   eval_metric='auc',
                   eval_set=[(X_test, y_test)],
                   verbose=False)
print(optimal_params.best_params_)


# In[71]:


# The previous params are confirmed
model_1 = xgb.XGBClassifier(objective='binary:logistic', gamma=0.25, learn_rate=0.05,
                        max_depth=5, reg_lambda=10, scale_pos_weight=2, n_estimators=10000, n_jobs=10)
model_1.fit(X_train, y_train, verbose=True,
            early_stopping_rounds=10, eval_metric='auc', eval_set=[(X_test, y_test)])


# In[72]:


y_pred = model.predict(X_test)
a = {'AUC_ROC' : roc_auc_score(y_test, y_pred), 'Precision' : precision_score(y_test, y_pred),
        'Recall' : recall_score(y_test, y_pred)}

results = pd.concat([results, pd.Series(a, name='kNNImput_Stratify_optimized')], axis=1)


# In[73]:


print(confusion_matrix(y_test, y_pred)/y_pred.shape)
print('Area under Roc curve:', roc_auc_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))


# ## Ensambled Resampling

# In[74]:


a = ensemble_xgboost_CV(X_y, threads=10)


# In[75]:


results = pd.concat([results, pd.Series(a, name='kNNImput_Ensembled')], axis=1)


# # Confrontation

# In[78]:


results


# # Final Considerations 
# Stratifying per se yelds worse perfomance in terms of area under the ROC curve than ensembled resampling, while, through parameter optimization, it gives the best results overall.
# Leaving the missing values handling to xgboost makes no particular difference with imputing them with kNN. The latter performs slightly better, but the fact that some values have been imputed by an algorithm could be easily exploited by xgboost since imputation does not come from chance. 
# Generally speaking, Ensembled samples, as a way to fight imbalance, give good results but are more prone to enhance recall rater than precision.
