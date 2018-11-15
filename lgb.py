# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 20:22:57 2018
lightgbm 试一下
@author: petter
"""

# pylint: disable = invalid-name, C0111
import lightgbm as lgb
import pandas as pd
#from sklearn.metrics import mean_squared_error
#from sklearn.preprocessing import OneHotEncoder

print('Loading data...')
# load or create your dataset
df_train = pd.read_csv('operation_train_new.csv')
df_test = pd.read_csv('operation_round1_new.csv')
df_label = pd.read_csv('tag_train_new.csv')


df_train=pd.merge(df_train,df_label,on="UID",how="inner")

drop_future=[
        "wifi",
        "geo_code",
        'ip1_sub',
        'ip2_sub',
        'ip2',
        'ip1',
        'mac2',
        'mac1',
        'device_code3',
        'device_code2',
        'device_code1',
        'device2',
        'device1',
        'mode',
        'time',
        'version'
              ]

X_train = df_train.drop(drop_future, axis=1)
y_train = df_train['Tag']
X_test = df_test.drop(drop_future, axis=1)

#X_train['time']=X_train['time'].astype(float)
#print(X_train.head)

X_test=X_test.fillna(0)
X_train=X_train.fillna(0)
#onehot coding
'''
print('onehot ++++')
enc = OneHotEncoder(handle_unknown='ignore')

X_train=enc.fit(X_train)
X_test=enc.fit(X_test)
'''
print("1:",len(X_train))
print("2:",len(y_train))



# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
#lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'l1'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

print('Starting training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=20,
                early_stopping_rounds=5)

print('Saving model...')
# save model to file
gbm.save_model('model.txt')

print('Starting predicting...')
# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
# eval
print(y_pred)
#print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)