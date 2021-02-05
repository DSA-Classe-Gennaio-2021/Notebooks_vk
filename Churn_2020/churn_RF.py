import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('train.csv')

#checking for missing data
data.isnull().values.any()

plt.hist(data['churn']);

# We now divide the df in categorical (encoding it) and continuos to check dependence
# between the features and the output

cat = data[['state','area_code','international_plan',
            'voice_mail_plan','churn']]
cont = data.drop(['state','area_code','international_plan',
            'voice_mail_plan','churn'], axis=1)

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
cat = cat.apply(LabelEncoder().fit_transform)

#Add econoded churn to continuos dataset
cont['churn'] = cat['churn']

# chi squared and ANOVA work great for binary output and for categorical
# and continous features respectively

from sklearn.feature_selection import chi2, f_classif

chi_scores = chi2(cat.drop(['churn'], axis=1), cat['churn'])

p_values = pd.Series(chi_scores[1],index = cat.columns[0:-1])
p_values.plot.bar())

anova_scores = f_classif(cont.drop(['churn'], axis=1), cont['churn'])

f_p_values = pd.Series(anova_scores[1],index = cont.columns[0:-1])
f_p_values.plot.bar();

#Let's drop columns accordingly and make the test df
cont= cont.drop(['account_length','total_day_calls','total_eve_calls', 'total_night_calls'], axis=1)

processed_data = pd.concat([cat[['international_plan','voice_mail_plan']],cont], axis=1)

ch1_rows = processed_data[processed_data['churn'] == 1].shape[0]
ch0_rows = processed_data[processed_data['churn'] == 0].shape[0]

resample_size_rare, resample_size_abund = 59 , 365

# Defining functions to implement ensemble sampling with RF

#Returns sampled datasets

def get_resampled_datasets(abundant, rare):
    iters = round(abundant.shape[0]/rare.shape[0])
    splits = np.array_split(abundant, iters)
    datasets = []
    for i in range(iters):
        datasets.append(pd.concat([splits[i],rare], axis=0))
    return datasets


# Returns prediction from different models

def get_RF_ensemble_predictions(datasets, test):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    X_tst = test.drop('churn', axis=1)

    predictions = []

    for k in range(len(datasets)):
        X = datasets[k].drop('churn', axis=1)
        y = datasets[k]['churn']
        model.fit(X, y)
        predictions.append(model.predict(X_tst))

    return predictions

#Scaling is not necessary since Random Forest is not affected by the scale of the features.
#Outlier detection and removal will be implemented in later versions
from sklearn.utils import shuffle
churn_0 = processed_data[processed_data['churn'] == 0].reset_index().drop('index',axis=1)
churn_1 = processed_data[processed_data['churn'] == 1].reset_index().drop('index',axis=1)

from sklearn.metrics import accuracy_score

# implementing cross-validation

for i in range(int(ch1_rows / resample_size_rare)):
    if i == int(ch1_rows / resample_size_rare) - 1:  # at last iteration we take all the remaining data
        ch0_lower, ch0_upper = i * resample_size_abund, (
                    i + 2) * resample_size_abund  # define were to split data for cross validation
        ch1_lower, ch1_upper = i * resample_size_rare, (i + 2) * resample_size_rare
    else:
        ch0_lower, ch0_upper = i * resample_size_abund, (i + 1) * resample_size_abund
        ch1_lower, ch1_upper = i * resample_size_rare, (i + 1) * resample_size_rare

    ch0_test = churn_0.iloc[ch0_lower:ch0_upper, :]
    ch1_test = churn_1.iloc[ch1_lower:ch1_upper, :]

    test = shuffle(pd.concat([ch0_test, ch1_test], axis=0))

    ch0_train = churn_0[~churn_0.isin(ch0_test)].dropna()
    ch1_train = churn_1[~churn_1.isin(ch1_test)].dropna()

    datafs = get_resampled_datasets(ch0_train, ch1_train)  # get n datasets for ensembled resampling
    pred = get_RF_ensemble_predictions(datafs, test)  # get n predictions for n models trained on n datasets
    cum_pred = np.round(np.mean(np.array(pred), axis=0))  # picking the modal class (0 or 1) and get unique prediction

    print(accuracy_score(test['churn'], cum_pred))  # testing prediction accuracies