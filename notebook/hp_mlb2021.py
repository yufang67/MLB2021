import numpy as np
import pandas as pd
#import shap
import joblib
import time
#import argparse
#import shap
from pathlib import Path
from functools import partial

#from sklearn.model_selection import train_test_split
#from sklearn.model_selection import StratifiedShuffleSplit
#from sklearn.preprocessing import RobustScaler,Normalizer,QuantileTransformer,PowerTransformer,StandardScaler
#from sklearn.preprocessing import OrdinalEncoder,LabelEncoder,OneHotEncoder
from sklearn.metrics import mean_squared_error,r2_score,mean_squared_log_error,mean_absolute_error
#from sklearn.metrics import make_scorer
#from sklearn.model_selection import cross_val_score,cross_validate

#import xgboost as xgb
import lightgbm as lgb
#from catboost import CatBoostRegressor
#from fastai.tabular.all import *
import optuna

import warnings
warnings.filterwarnings(action="ignore")

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
np.set_printoptions(threshold=1000)
pd.options.display.max_info_columns=1000

DATA_PATH = Path('../data')


param_lgb = {
    'device_type':'gpu',
    'gpu_device_id':1,
    'gpu_platform_id': 0,
    #'gpu_use_dp': False,
    'n_jobs' : 2,
    'seed':42,
    
    }
    
def optimize(trial,x,y,testx,testy):
    
    #metric=[]
    #cv = KFold(n_splits=5,shuffle=True,random_state=42)
    
    param = {
    #'num_iterations': 10000, 
    'early_stopping_round': 100,
    'seed':42,
    'n_jobs' : 2,
    #'verbose_eval' : False,
    #'verbose': 10000,
    #'gpu_use_dp': False,
    'device_type': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 1,
    #'boosting_type': 'gbdt',
    #'objective': 'mse',
    'objective': trial.suggest_categorical('objective',['mse','mae','huber']),
    'learning_rate' : trial.suggest_loguniform('learning_rate', 1e-4,1e-1),
    'n_estimators' : trial.suggest_int('n_estimators',100,5000),
    'max_depth' : trial.suggest_int('max_depth', 3, 20),
    'num_leaves' : trial.suggest_int('num_leaves', 2, 2**17),
    'max_bin' : trial.suggest_int('max_bin', 10, 250), 
    'feature_fraction' : trial.suggest_uniform('feature_fraction', 0.1, 1),
    'bagging_fraction' : trial.suggest_uniform('bagging_fraction', 0.1, 1),    
    'bagging_freq' : trial.suggest_int('bagging_freq', 1, 100),
    'min_sum_hessian_in_leaf' : trial.suggest_int('min_sum_hessian_in_leaf', 1, 10),
    'reg_alpha' : trial.suggest_loguniform('reg_alpha', 1e-5, 1e-1),
    'reg_lambda' : trial.suggest_loguniform('reg_lambda', 1e-5, 1e-1),
}

    ml = lgb.LGBMRegressor(**param)
    ml.fit(x,y,eval_set=[(testx, testy)],verbose=10000)
    val_pred  = ml.predict(testx)

    # metric
    metric = mean_absolute_error(testy,val_pred)
    return metric

col_key = ['date','date_playerId']
col_label = ['target1','target2','target3','target4']


if __name__ == '__main__':

    #train = pd.read_csv(DATA_PATH/'df_targ2_gtr.csv')
    #test = pd.read_csv(DATA_PATH/'df_targ2_gtt.csv')
    
    train = pd.read_csv(DATA_PATH/'df_targ2_ntr.csv')
    test = pd.read_csv(DATA_PATH/'df_targ2_ntt.csv')
    

    trainX = train.drop(columns=col_key+col_label)
    trainy = train['target2']
    testX = test.drop(columns=col_key+col_label)
    testy = test['target2']


    optimization_function = partial(
        optimize, x=trainX.values, y=trainy.values,
        testx=testX.values,testy=testy.values
        )
    
    study = optuna.create_study(direction="minimize")

    start = time.time()
    study.optimize(optimization_function,n_trials=100)
    end = time.time()

    print('Time consumed (min): {:.1f}'.format((end-start)/60))
    print('Best trial: ')
    print(study.best_params)


