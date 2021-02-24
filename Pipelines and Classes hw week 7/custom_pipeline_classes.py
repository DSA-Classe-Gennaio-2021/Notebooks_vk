import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import stats


# credits to: https://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn
# for the below class which was written from ispiration

class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self,columns = None, keep_nan=False):
        self.columns = columns # array of column names to encode
        self.keep_nan = keep_nan

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X, y=None):
      
        output = X.copy()
        # if NaNs are not in dataset we just label encode each specified column
        # or all of them if not specified
        if self.keep_nan == False:
            if self.columns is not None:
                for col in self.columns:
                    output[col] = LabelEncoder().fit_transform(output[col])
            else:
                for colname,col in output.iteritems():
                    output[colname] = LabelEncoder().fit_transform(col)
            
            #If y is passed, it is encoded as well
            if y is not None:
                target =  LabelEncoder().fit_transform(y)
            
                return output, target
            else:
                return output
            
        # else we will use masking to keep track of NaNs and restore them after label encoding
        else:
            if self.columns is not None:
                for col in self.columns:
                    original = output[col]
                    mask = output[col].isnull()
                    new_col = LabelEncoder().fit_transform(output[col].astype('str'))
                    new_col = new_col.astype('int')
                    new_col = pd.Series(new_col, name = output[col].name)
                    output[col] = new_col.where(~mask, original)
            else:
                for colname,col in output.iteritems():
                    original = output[colname]
                    mask = output[colname].isnull()
                    new_col = LabelEncoder().fit_transform(output[colname].astype('str'))
                    new_col = new_col.astype('int')
                    new_col = pd.Series(new_col, name = colname)
                    output[colname] = new_col.where(~mask, original)
            
            #If y is passed, it is encoded as well
            if y is not None:
                original = y
                mask = y.isnull()
                target =  LabelEncoder().fit_transform(y.astype('str'))
                target = target.astype('int')
                target = pd.Series(target, name = y.name)
                target = target.where(~mask, original) 
                    
                return output, target 
            else:
                return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X, y)
    
    

#Class to one hot encode multiple columns eventually keeping NaNs

class Robust_OneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns, keep_nan=False):
        self.columns = columns # array of column names to encode
        self.keep_nan = keep_nan

    def fit(self, X, y=None):
        return self # not relevant here

    def transform(self, X, y=None):
        
        output = X.copy()
        
        for col in self.columns:   #create a df with encoded columns for each category in col
            
            df = pd.get_dummies(X[col])  # Encoding
            
            if self.keep_nan == True:  #restoring NaNs
                for index, row in df.iterrows():
                    # if there are all zero rows it means Nan for those features
                    if pd.Series(row.values).any():
                        pass
                    else:
                        # replacing zeros and restoring Nans
                        new_row = row.replace(0, np.nan, inplace=True)  
                        df.loc[index] = new_row
                
            # Changing columns labels to get unique for each one
            same_columns_names = set(df.columns).intersection(output.columns)
            if same_columns_names:
                count = 1
                new_names = []
                for i in range(len(df.columns)):
                    name = df.columns[i]                        
                    new_names.append(f'{name}_{count}')
                count += 1
                df.columns = new_names
        
        #To this point we should have got a df with encoded columns to concatenate to output
        output = pd.concat([output,df], axis=1)
        
        # Now we drop the old columns    
        output.drop(self.columns, axis=1, inplace=True)
        
        
        #If y is passed, it is encoded as well
        if y is not None and self.keep_nan == True:
            original = y
            mask = y.isnull()
            target =  LabelEncoder().fit_transform(y.astype('str'))
            target = target.astype('int')
            target = pd.Series(target, name = y.name)
            target = target.where(~mask, original) 
            
            return output, target 
            
            
        elif y is not None and self.keep_nan == False:
            target =  LabelEncoder().fit_transform(y)
            
            return output, target 
            
        else:
            return output       

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
    
 
#
# class used to encode both ordinal and non-ordinal categorical data
# with the chance to keep NaN's if needed

class Whole_MultiEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, non_ordinal = None , ordinal = None, keep_nan=False):         
        self.non_ordinal = non_ordinal 
        self.ordinal = ordinal
        self.keep_nan = keep_nan

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self, X, y=None):
        
        
        
        
        if y is not None:
            skip_y = False
            if self.ordinal:
                
                # Little implementation to make it work after feature selection
                self.ordinal = list(set(X.columns).intersection(self.ordinal))
                
                X, y = MultiColumnLabelEncoder(columns=self.ordinal,
                                               keep_nan=self.keep_nan).fit_transform(X, y)
                self.keep_nan = False
                skip_y = True
                
            if self.non_ordinal:
                
                # Little implementation to make it work after feature selection
                self.non_ordinal = list(set(X.columns).intersection(self.non_ordinal))
                
                if skip_y == True:
                    X = Robust_OneHotEncoder(columns=self.non_ordinal, keep_nan=self.keep_nan).fit_transform(X)
                    return X, y
                else:
                    X, y = Robust_OneHotEncoder(columns=self.non_ordinal, keep_nan=self.keep_nan).fit_transform(X, y)
                    return X, y
                
        else:
            if self.ordinal:
                self.ordinal = list(set(X.columns).intersection(self.ordinal))
                X = MultiColumnLabelEncoder(columns=self.ordinal,
                                               keep_nan=self.keep_nan).fit_transform(X)
                return X
            
            if self.non_ordinal:
                self.non_ordinal = list(set(X.columns).intersection(self.non_ordinal))
                X = Robust_OneHotEncoder(columns=self.non_ordinal, keep_nan=self.keep_nan).fit_transform(X)
                
                return X
            

       
    

# Custom Scaling class.. the aim is to make sure that Standard Scaling is applied to only continous variables
# making, in addition to it, outlier  detection via z-score applicable 

class custom_scale(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        df = X.copy()
        
        scaler = StandardScaler(with_mean=True, with_std=True)
        df[self.columns] = scaler.fit_transform(df[self.columns])
        
        if y is not None:
            return df, y
        else:
            return df
            

    
    
# Custom outlier detector and remover so only continous features are processed

class custom_outlier_Zscore(BaseEstimator, TransformerMixin):
    def __init__(self, columns, score=3):
        self.columns = columns
        self.score = score
        
    
    def fit(self, X, y=None):
        df = X[self.columns].copy()
        z = np.abs(stats.zscore(df))
        rows_to_remove = np.where(z > self.score)
        self.rows_to_remove = rows_to_remove[0]
        
        return self
   
    def transform(self, X, y=None):
        df = X.copy()
        remove = self.rows_to_remove
        df = df.drop(remove, axis=0)
        
        if y is not None:
            target = y.drop(remove)
            return df, target
        else:
            return df
        
        
     
    