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


class Process_management(object):
    def datasource_reader(self, file_path, index_name='id', train_suffix='train.csv', \
        test_suffix='test.csv',label_suffix='label.csv',):
        ##Read datasets traversingly, datasets saved 
        ## with format of 'id-features', target saved with 'id-tgt', 
        ## merge with reduce function.
        df_trains = []
        df_tests = []
        df_trains_path = []
        df_tests_path = []
        for roots, dirs, files in os.walk(file_path):
            for name in files:
                if name.endswith(label_suffix):
                    df_label = pd.read_csv(file_path+name)
                elif name.endswith(train_suffix):
                    df_trains_path.append((file_path+name,name))
                elif name.endswith(test_suffix):
                    df_tests_path.append((file_path+name,name))

        for (path,name) in df_trains_path:
            df_trains.append(pd.read_csv(path))
            df_trains[-1].rename(index=str,columns={x: x+'_'+name.split('_',1)[0] \
            for x in df_trains[-1].columns.tolist() if x not in [index_name]})
        for (path,name) in df_tests_path:
            df_tests.append(pd.read_csv(path))
            df_tests[-1].rename(index=str,columns={x: x+'_'+name.split('_',1)[0] \
            for x in df_tests[-1].columns.tolist() if x not in [index_name]})
        df_trains = reduce(lambda df1,df2: \
        pd.merge(df1,df2,on=[index_name],how='inner'),df_trains)
        df_trains = pd.merge(df_trains,df_label,on=[index_name],how='inner')

        df_tests = reduce(lambda df1,df2: \
        pd.merge(df1,df2,on=[index_name],how='inner'),df_tests)

        return(df_trains, df_tests)

        
        
            