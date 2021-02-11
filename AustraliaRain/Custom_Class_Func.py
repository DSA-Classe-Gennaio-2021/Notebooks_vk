import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# Defining functions to implement ensemble sampling

#Returns sampled datasets

def get_resampled_datasets(abundant, rare):
    iters = round(abundant.shape[0]/rare.shape[0])
    splits = np.array_split(abundant, iters)
    datasets = []
    for i in range(iters):
        datasets.append(pd.concat([splits[i],rare], axis=0))
    return datasets


# Returns prediction from different models

def get_xgboost_ensemble_predictions(datasets, test, jobs=None):
    import xgboost as xgb
    model = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, n_estimators=10000, n_jobs=10)
    X_tst = test.iloc[:, 0:-1]
    y_tst = test.iloc[:, -1]

    predictions = []

    for k in range(len(datasets)):
        X = datasets[k].iloc[:, 0:-1]
        y = datasets[k].iloc[:, -1]
        model.fit(X, y, verbose=False,
            early_stopping_rounds=10, eval_metric='aucpr', eval_set=[(X_tst, y_tst)])
        predictions.append(model.predict(X_tst))
        
    unique_pred = np.round(np.mean(np.array(predictions), axis=0))

    return unique_pred

# Defining a custom CV function to use with the previous custom functions.

def ensemble_xgboost_CV(dataset, cv=5, threads=None):  #threads --> n_jobs for xgboost 
    from sklearn.utils import shuffle
    from sklearn.metrics import roc_auc_score , precision_score, recall_score
    from sklearn.metrics import confusion_matrix
    import numpy as np
    
    df0 = dataset[dataset.iloc[:, -1] == 0].reset_index().drop('index',axis=1)
    df1 = dataset[dataset.iloc[:, -1] == 1].reset_index().drop('index',axis=1)
    
    if df0.shape[0] < df1.shape[0]:      #defining abundant and rare samples
        rare = df0
        abund = df1
    else:
        rare = df1
        abund = df0
    
    resample_size_abund = int(abund.shape[0]/cv)
    resample_size_rare = int(rare.shape[0]/cv)
    
    # Implementing a Cross-Validation
    
    results = []
    
    for i in range(cv):
        if i == int(cv) -1:  #at last iteration we take all the remaining data
            abund_lower, abund_upper = i*resample_size_abund, (i+2)*resample_size_abund  #define were to split data for cross validation
            rare_lower, rare_upper = i*resample_size_rare, (i+2)*resample_size_rare
        else:
            abund_lower, abund_upper = i*resample_size_abund, (i+1)*resample_size_abund
            rare_lower, rare_upper = i*resample_size_rare, (i+1)*resample_size_rare
        
        abund_test = abund.iloc[abund_lower:abund_upper,:]
        rare_test = rare.iloc[rare_lower:rare_upper,:]
        

        test = shuffle(pd.concat([abund_test,rare_test], axis=0))
        y_test = test.iloc[:,-1]
                
        abund_train = abund.drop(abund_test.index)
        rare_train = rare.drop(rare_test.index)        

        datafs = get_resampled_datasets(abund_train, rare_train) #get n datasets for ensembled resampling        
        pred = get_xgboost_ensemble_predictions(datafs, test, jobs=threads)  #get unique prediction for n models trained on n datasets
        
        roc = roc_auc_score(y_test, pred)
        preci = precision_score(y_test, pred)
        recall = recall_score(y_test, pred)
        confusion_m = confusion_matrix(y_test, pred)/pred.shape
        
        results.append([roc, preci, recall])
        
        print(confusion_m)
        print('Area under Roc curve:', roc )
        print('Precision:', preci)
        print('Recall:', recall)
        print("\n")
    
    results = np.mean(np.array(results), axis=0)
    
    print('Average scores:', '\n')
    print('Area under Roc curve:', results[0] )
    print('Precision:', results[1])
    print('Recall:', results[2])
    
    return {'AUC_ROC' : results[0], 'Precision' : results[1], 'Recall' : results[2]}

    


# credits to: https://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn
# for the below class

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)