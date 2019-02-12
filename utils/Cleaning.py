import os
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
from sklearn.externals import joblib
import logging
import os
import random
import sys
import multiprocessing as mp
from functools import reduce

import glob
from tqdm import tqdm
import yaml
#from attrdict import AttrDict

nowpath = os.path.abspath('.')


class Data_Cleaner(object):
    def cat_col_detector(self, df, in_cols, thre, df_test=None): 
        ## To detect categorical columns, if unique>thre, define as categorical.
        cat_cols = []
        cat_col_cats = {}
        ## categorical columns and how many cats in each col
        for i in in_cols:
            num_cats = len(df[i].unique())
            if num_cats < thre:
                cat_cols.append(i)
                cat_col_cats[i] = num_cats
        return(cat_cols, cat_col_cats)

    def nan_dummy_getter(self, df, in_cols, df_test=None):
        ## Get dummies for Nan, out_cols to record dummy columns
        out_cols = []
        for i in in_cols:
            if (df[i].isnull().any().any() or df_test[i].isnull().any().any()):
                df[i+'_isnan'] = df[i].isnull().astype('int')
                if df_test is not None:
                    df_test[i+'_isnan'] = df_test[i].isnull().astype('int')
                out_cols.append(i+'_isnan')
        return(df, df_test, out_cols)

    def categorical_encoder(self, df, in_cols, df_test=None):
        ## Convert a list of categorical columns into int, range(len(unique))
        for col in in_cols:
            value_list = df[col].unique().tolist()
            df[col] = df[col].apply(lambda x: value_list.index(x))
            if df_test is not None:
                df_test[col] = df_test[col].apply(lambda x: value_list.index(x))
        cat_cols = in_cols
        return(df, df_test, cat_cols)
    ##!!!!!!!!!!!Add infinity handeler
    
    def dummy_getter(self, df, in_cols, drop_first=True, dummy_na=True, df_test=None):
        ## get dummy, one dummy for Nan and drop first dummies
        if df_test is not None:
            tem = pd.concat([df[in_cols],df_test[in_cols]], axis=0)
        elif df_test is None:
            tem = pd.DataFrame(df[in_cols])
        tem_dummies = pd.get_dummies(tem.astype(str), prefix=in_cols, \
                                    drop_first=drop_first, dummy_na=dummy_na)
        out_cols = tem_dummies.columns.tolist()
        df = df.drop(in_cols, axis=1)
        df = pd.concat([df, tem_dummies.iloc[range(df.shape[0])]], axis=1)
        if df_test is not None:
            df_test = df_test.drop(in_cols, axis=1)
            df_test = pd.concat([df_test, tem_dummies.iloc \
            [df.shape[0]:tem_dummies.shape[0]]], axis=1)          
        return(df,df_test,out_cols)

    def nan_filler(self, df, in_cols, method='median', df_test=None):
        ## !!!!!!!!!!!!!!!Categorical
        ## fillin Na with mean/median/-999/zero
        if type(method) != type('str'):
            switcher = {method: (lambda df,col: method)}
        else:
            switcher = {
                'mean': (lambda df,col: df[col].mean()),
                'median': (lambda df,col: df[col].median()),
                method: (lambda df,col: eval('np.{}'.format(method))(df[col]))
            }
        for i in in_cols:
            df[i] = df[i].fillna(switcher[method](df,i))
            if df_test is not None:
                df_test[i] = df_test[i].fillna(switcher[method](df,i))
        return(df,df_test)

    def normalizer(self, df, in_cols, method='median', measure='range', \
                    df_test=None, inplace=True):
        ## Normalization, keep consistency between train & test
        switcher = {
            'median': (lambda df,col: df[col].median()),
            'mean': (lambda df,col: df[col].mean()),
            method: (lambda df,col: eval('np.{}'.format(method))(df[col]))
        }
        ###
        for i in in_cols:
            referer = switcher[method](df,i)
            maxer = df[i].max()
            miner = df[i].min()
            if maxer == miner:
                print('Const value in '+str(i)+' !!!')
                continue
            if measure == 'range':
                denom = maxer-miner
            elif measure == 'std':
                denom = df[i].std()
            if inplace == False:
                df[i+'_norm'] = (df[i] - referer)/denom
            else:
                df[i] = (df[i] - referer)/denom
            if df_test is not None:
                if inplace == False:
                    df_test[i+'_norm'] = (df_test[i] - referer)/denom
                else:
                    df_test[i] = (df_test[i] - referer)/denom
        return(df,df_test)





