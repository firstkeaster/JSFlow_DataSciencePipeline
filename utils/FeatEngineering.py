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

class Aggre_based_FE(object):
    def basic_aggregation(self, df, groupby_cols, stat_cols, methods=['mean'], \
                          df_test=None, delta=False):
        ##Perform basic aggregation on specified columns,
        ## columns to be stated and method.
        ## methods: max, min, mean, sum, var, median
        ## delta == True: do subtraction between parent and son cols
        group_obj = df.groupby(groupby_cols)
        new_cols = []
        parents_cols = {}
        for col in stat_cols:
            for method in methods:
                the_new_col = '{}_{}_{}'.format('_'.join(groupby_cols),method,col)
                df = df.merge(group_obj[col].agg(method).reset_index(). \
                            rename(index=str,columns={col:the_new_col}) \
                            [groupby_cols + [the_new_col]],on=groupby_cols, \
                            how='left')
                if df_test is not None:
                    df_test = df_test.merge(group_obj[col].agg(method).reset_index().\
                            rename(index=str,columns={col:the_new_col})\
                            [groupby_cols + [the_new_col]],on=groupby_cols,\
                            how='left')
                parents_cols[the_new_col] = col

                new_cols.append(the_new_col)
        if delta == True:
            (df,df_test,new_cols_delta,parents_cols_delta) = \
            self.basic_func(df,parents_cols,df_test)
            
            new_cols.extend(new_cols_delta)
            parents_cols.update(parents_cols_delta)

        return(df,df_test,new_cols,parents_cols)

    #def basic_delta(self,df,in_cols,df_test=None):
    def basic_func(self,df,in_cols,func='-',df_test=None):
        ##apply function!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        ##Get the deltas/functions between oritinal value and aggregation
        ##in_cols{subtraction:[subtracted]}
        new_cols = []
        parents_cols = {}
        if func in ['+','-','*','/']:
            operator = (lambda x1,x2:eval('x1'+func+'x2'))
        else:
            operator = func
        for r_col in in_cols:
            for l_col in in_cols[r_col]:
                the_new_col = '{}_{}_delta'.format(l_col,r_col)
                df[the_new_col] = operator(df[l_col],df[r_col])
                if df_test is not None:
                    df_test[the_new_col] = operator(df_test[l_col],df_test[r_col])

                new_cols.append(the_new_col)
                parents_cols[the_new_col] = [l_col,r_col]

        return(df,df_test,new_cols,parents_cols)
    
    def __group_applier(self, x, groupby_cols, crit_cols, stat_cols, crit_func=False, \
                        func_name=False, methods=['max'], magics=False):
        candidates = {}
        for crit_col in crit_cols:
            for stat_col in stat_cols:
                if magics is not False:
                    for method in methods:
                        candidates[crit_col+'_'+func_name+'_'+stat_col+'_'+method] = \
                        x.loc[crit_func(x[crit_col]),stat_col].agg(method)
                else:
                    ##Try magic
                    for (magnum, mag) in enumerate(magics):
                        candidates[crit_col+'_'+func_name+'_'+stat_col+'_'+mag] = \
                        eval('x.loc[crit_func(x[crit_col]),stat_col]'+mag)
        return pd.DataFrame({col: value for (col, value) in candidates.items()
                            }.update({col: x[col].unique() for col in groupby_cols}),
                            index = '1')

    def cond_aggregation(self, df, groupby_cols, crit_cols, stat_cols, crit_func=False, \
                        func_name=False, methods=['max'], magics=False, \
                        df_test=None):
        ## Conditional aggregation, apply specified function to groupby items.
        ## Example: groupby id, for each id, multiple records are
        ## like this:{'Status':[0,0,0,1,1,0],'Months_Balance':[34,24,45,23,1,0]}.
        ## We need the maximum Months_Balance when status is 0 as a feature for this id.
        ## Then ['id'] as groupby_cols, ['Status'] as crit_cols, ['Months_balance'] as stat_cols,
        ## crit_func as (lambda x: x['Status'] == 0), methods=['max'], with no magics.
        ##
        ## crit_func: must satisfy use one input and give back a bool output
        if crit_func == False:
            crit_func = (lambda x: x < 0)
            func_name = 'BLZero'
            
        n_df = df.groupby(groupby_cols, as_index=False).apply(self.__group_applier, \
        groupby_cols=groupby_cols, crit_cols=crit_cols, stat_cols=stat_cols, \
        crit_func=crit_func, func_name=func_name, methods=methods, magics=magics
        )
        if df_test is not None:
            n_df_test = df_test.groupby(groupby_cols, as_index=False).apply(self.__group_applier, \
            groupby_cols=groupby_cols, crit_cols=crit_cols, stat_cols=stat_cols, \
            crit_func=crit_func, func_name=func_name, methods=methods, magics=magics
            )

        n_df = n_df.reset_index().drop('index',axis=1)
        n_df_test = n_df_test.reset_index().drop('index',axis=1)
        return(n_df,n_df_test)


        



class Parallelers_FE(object):
    def __chunk_groups(self, groupby_object, chunk_size):
        ##yield: return index_chunk_, group_chunk_ one by one, 
        ## you can call chunk_groups as a iterator.
        n_groups = groupby_object.ngroups
        group_chunk, index_chunk = [], []
        for i, (index, df) in enumerate(groupby_object):
            group_chunk.append(df)
            index_chunk.append(index)

            if (i + 1) % chunk_size == 0 or i + 1 == n_groups:
                group_chunk_, index_chunk_ = group_chunk.copy(), index_chunk.copy()
                group_chunk, index_chunk = [], []
                yield(index_chunk_, group_chunk_)


    def group_paralleler(self, groupby_object, func, index_name='id', \
        num_workers=1, chunk_size=10000):
        ## index: index of groupby object
        n_chunks = np.ceil(1.0 * groupby_object.ngroups / chunk_size)
        indeces, new_feat = [], []
        for index_chunk, groups_chunk in tqdm(self.__chunk_groups(groupby_object, chunk_size), \
        total=n_chunks):
            with mp.pool.Pool(num_workers) as executor:
                new_feat_chunk = executor.map(func, groups_chunk)
            new_feat.extend(new_feat_chunk)
            indeces.extend(index_chunk)

        new_feat = pd.DataFrame(new_feat)
        new_feat.index = indeces
        new_feat.index.name = index_name
        return(new_feat)
