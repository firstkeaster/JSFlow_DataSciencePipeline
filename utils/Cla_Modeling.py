import os
import timeit
from datetime import date
from datetime import time
from datetime import datetime
from datetime import timedelta
from pandas import Series, DataFrame
import pickle as pk
import pandas as pd
import numpy as np

from sklearn import metrics
from numpy import loadtxt

from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.preprocessing import LabelEncoder

from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgb

# import File_manager
import json


class Classifiers(object):
    def __init__(self, file_path):
        self.file_path = file_path
        if not os.path.exists(file_path+'/documents/model_parameters.json'):
            with open(file_path+'/documents/model_parameters.json','w') as pf:
                self.params = {}
                self.params['lightGBM'] = {}
                self.params['lightGBM']['1'] = {'learning_rate': 0.01, 
                                "objective": "binary", 
                                'max_depth': 8,  
                                'min_child_samples': 50, 
                                'max_bin': 100,  
                                'subsample': 0.8,  
                                'subsample_freq': 1,  
                                'subsample_for_bin': 20000,  
                                'colsample_bytree': 0.632,  
                                'min_child_weight': 19,  
                                'scale_pos_weight':99, 
                                'is_unbalanced': True,
                                'metric': 'accuracy'
                                }
                json.dump(self.params, pf)
                pf.close()
        else:
            with open(file_path+'/documents/model_parameters.json','w') as pf:
                self.params = json.load(pf)
                pf.close()
        if not os.path.exists(file_path+'/documents/model_settings.json'):
            with open(file_path+'/documents/model_settings.json','w') as pf:
                self.setts = {}
                self.setts['lightGBM'] = {}
                self.setts['lightGBM']['1'] = {'valid_names': ['train', 'valid'],
                                'evals_results': {},
                                'early_stopping_rounds': 300,
                                'num_boost_round': 10000,
                                'verbose_eval': 10,
                                'feval': None,
                                'learning_rates': lambda iter: \
                                np.log10(max(2,10002-iter))/400
                                }
                json.dump(self.setts, pf)
                pf.close()
        else:
            with open(file_path+'/documents/model_settings.json','w') as pf:
                self.setts = json.load(pf)
                pf.close()
        if not os.path.exists(file_path+'/documents/model_features.json'):
            with open(file_path+'/documents/model_features.json','w') as pf:
                self.feats = {}
                self.feats['lightGBM'] = {}
                self.feats['lightGBM']['1'] = {'predictors': [],
                                'categorical_features': [],
                                'label': [],
                                }
                json.dump(self.feats, pf)
                pf.close()
        else:
            with open(file_path+'/documents/model_features.json','w') as pf:
                self.feats = json.load(pf)
                pf.close()
    
    def add_params(self, model='lightGBM', \
                    name=str(np.random.randint(100,999)), params={}):
        self.params[model][name] = params
        with open(self.file_path+'/documents/model_parameters.json','w') as pf:
            json.dump(self.params, pf)
            pf.close()
        return('Param setting name: '+name)

    def add_settings(self, model='lightGBM', \
                    name=str(np.random.randint(100,999)), params={}):
        self.setts[model][name] = params
        with open(self.file_path+'/documents/model_settings.json','w') as pf:
            json.dump(self.setts, pf)
            pf.close()
        return('Model setting name: '+name)

    def add_features(self, model='lightGBM', \
                    name=str(np.random.randint(100,999)), params={}):
        self.feats[model][name] = params
        with open(self.file_path+'/documents/model_features.json','w') as pf:
            json.dump(self.feats, pf)
            pf.close()
        return('Features setting name: '+name)

    def Classifier(self, train_data, train_label, val_data=None, \
                val_label=None, model='lightGBM', \
                params=None, setts=None, feats=None):
        if params is None:
            params = self.params[model]['1']
        if setts is None:
            setts = self.setts[model]['1']
        if feats is None:
            feats = self.feats[model]['1']
        switcher = {
            'lightGBM': self.__lightGBM_Builder,
            'XGBoost': self.__XGBoost_Builder,
            'RandomForest': self.__RandomForest_Builder,
            'NeuralNetwork': self.__NN_Builder
        }
        trained_model = switcher[model](train_data, train_label, \
                        val_data, val_label, params, setts, feats)
        return(trained_model)

    def __lightGBM_Builder(self, train_data, train_label, val_data=None, \
                val_label=None, params=None, setts=None, feats=None):

        train_lgbset = lgb.Dataset(train_data[feats['predictors']], \
                                label=train_label[feats['label']], \
                                feature_name=feats['predictors'], \
                                categorical_feature=feats['categorical_features'])
        
        if val_data is None:
            val_lgbset = None
            gbm = lgb.train(params, 
                                train_lgbset, 
                                evals_results=setts['evals_results'],
                                num_boost_round=setts['num_boost_round'],
                                verbose_eval=setts['verbose_eval'],
                                feval=setts['feval'],
                                learning_rates=setts['learning_rates'])
        else:
            val_lgbset = lgb.Dataset(val_data[feats['predictors']], \
                                label=val_label[feats['label']], \
                                featrue_name=feats['predictors'], \
                                categorical_feature=feats['categorical_features'])
            gbm = lgb.train(params, 
                                train_lgbset, 
                                valid_sets=[train_lgbset,val_lgbset],
                                valid_names=setts['valid_names'],
                                evals_results=setts['evals_results'],
                                num_boost_round=setts['num_boost_round'],
                                verbose_eval=setts['verbose_eval'],
                                feval=setts['feval'],
                                learning_rates=setts['learning_rates'])

        return(gbm)

    def __XGBoost_Builder(self, train_data, train_label, val_data=None, \
                val_label=None, params=None, setts=None, feats=None):
        ## to be constructed
        xgb_model = XGBClassifier(
                                learning_rate=params['learning_rate'],
                                n_estimators=params['n_estimators'],
                                max_depth=params['max_depth'],
                                min_child_weight=params['min_child_weight'],
                                gamma=params.get('gamma',0),
                                subsample=params.get('subsample',0.8),
                                colsample_bytree=params.get('colsample_bytree',0.8),
                                objective=params.get('objective','binary:logistic'),
                                tree_method=params.get('tree_method','gpu_hist'),
                                gpu_id=params.get('gpu_id','0'),
                                nthread=params.get('nthread',4),
                                scale_pos_weight=params.get('scale_pos_weight',1),
                                seed=params.get('seed',27))
        if val_data is None:
            xgb_model.fit(train_data, train_label)
        else:
            val_set = [(val_data, val_label)]
            xgb_model.fit(train_data, train_label, \
                        eval_metric=setts.get('eval_metric','error'), \
                        eval_set=val_set, verbose=setts.get('verbose',True))
        return(xgb_model)
    
    def __RandomForest_Builder(self, train_data, train_label, val_data=None, \
                val_label=None, params=None, setts=None, feats=None):
        ## to be constructed
        return

    def __NN_Builder(self, train_data, train_label, val_data=None, \
                val_label=None, params=None, setts=None, feats=None):
        ## to be constructed
        return

