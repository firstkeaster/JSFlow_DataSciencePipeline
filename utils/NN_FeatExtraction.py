# Author: Fu Shang <fu.shang@nyu.edu>
# License: 

import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc,recall_score,precision_score,roc_auc_score

import gc
import os
import time
import random
from sklearn.model_selection import train_test_split

from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.models import load_model
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek, SMOTEENN

import tensorflow as tf
from sklearn.metrics import roc_auc_score

from keras.layers import Dense, Activation, LSTM, Flatten, Dropout, Embedding, concatenate, Input
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras import optimizers, regularizers
from copy import deepcopy
import re
import io

class NN_RanSampNet(object):
    ## Should define methods including fit, evaluate, save, predict
    def __init__(self, net_width=20, subnet_width=10, hid_depth=2, \
                hid_decrement=0, conc_hid_depth=2, regularization=0.01, \
                learning_rate=0.001, beta_1=0.9,beta_2=0.999, \
                loss_func='binary_crossentropy',optimizer='sgd', \
                metrics=['accuracy']):
        self.model = None
        self.net_width = net_width
        self.subnet_width = subnet_width
        self.hid_depth = hid_depth
        self.hid_decrement = hid_decrement
        self.conc_hid_depth = conc_hid_depth
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.metrics = metrics

        self.regularization = regularization

        self.sam_cols = {}
        self.input_set = {}
        self.last_layers = {}
        self.dense_layers = {}
        self.all_layers = {}
        
        self.Layer_Den = [{} for i in range(self.net_width)]
        self.Layer_BatNorm = [{} for i in range(self.net_width)]
        self.Layer_Activ = [{} for i in range(self.net_width)]
        self.Layer_Dropot = [{} for i in range(self.net_width)]

        for i in range(self.net_width):
            self.input_set[i] = Input(\
                            shape=(self.subnet_width,),\
                            name='input_str(i)')
            self.Layer_Dropot[i][0] = Dropout(\
                            0.1)\
                            (self.input_set[i])
            self.Layer_Den[i][0] = Dense(\
                            self.subnet_width,\
                            name='L_Den_'+str(i)+'_0')\
                            (self.Layer_Dropot[i][0])
            self.Layer_BatNorm[i][0] = BatchNormalization(\
                            )\
                            (self.Layer_Den[i][0])
            self.Layer_Activ[i][0] = Activation(\
                            'tanh')\
                            (self.Layer_BatNorm[i][0])
            for j in range(1,self.hid_depth,1):
                self.Layer_Den[i][j] = Dense(\
                            self.subnet_width-j*self.hid_decrement,\
                            name='L_Den_'+str(i)+'_'+str(j))\
                            (self.Layer_Activ[i][j-1])
                self.Layer_BatNorm[i][j] = BatchNormalization(\
                            )\
                            (self.Layer_Den[i][j])
                self.Layer_Activ[i][j] = Activation(\
                            'relu')\
                            (self.Layer_BatNorm[i][j])
            self.last_layers[i] = Dense(\
                            self.subnet_width,\
                            activation='sigmoid',name='L_Den_'+\
                            str(i)+'_last')\
                            (self.Layer_Activ[i][self.hid_depth-1])
        self.all_layers[0] = concatenate(\
                            [self.last_layers[i] for i in \
                            range(self.net_width)],name='L_concated')

        for i in range(1,self.conc_hid_depth):
            self.all_layers[i] = Dense(\
                            self.net_width*self.subnet_width,\
                            activation='relu',\
                            kernel_regularizer=regularizers.l2(self.regularization),\
                            name='L_all_'+str(i))(self.all_layers[i-1])
        self.final_layer = Dense(\
                            1,activation='sigmoid',name='output_layer')\
                            (self.all_layers[self.conc_hid_depth-1])

        self.model = Model(\
                            inputs=[self.input_set[i] for i in range(self.net_width)],\
                            outputs=self.final_layer)
        self.adam = optimizers.Adam(\
                            lr=self.learning_rate,beta_1=self.beta_1,\
                            beta_2=self.beta_2)
        self.optimizer = self.adam
        self.model.compile(\
                            loss=self.loss_func,optimizer=self.optimizer, \
                            metrics=self.metrics)
                
        
    

    def fit(self, df_x, df_y, in_cols, id=['id'], epochs=10,\
            batch_size=10000, val_x=None, val_y=None):
        self.all_cols = in_cols
        for i in range(self.net_width):
            self.sam_cols[i] = np.random.choice(\
                            self.all_cols,\
                            self.subnet_width)
            self.input_set[i] = Input(\
                            shape=(self.subnet_width,),\
                            name="input_"+str(i))
        if val_x is None:
            validation_data = None
            eva = None
        else:
            validation_data = [[val_x[self.sam_cols[i]] for i in \
                            range(self.net_width)],val_y]
            eva = True
        self.results = self.model.fit(\
                            [df_x[self.sam_cols[i]] for i in \
                            range(self.net_width)],\
                            df_y,\
                            epochs=epochs,\
                            batch_size=batch_size,\
                            validation_data=validation_data)     
        if eva is not None:
            eva = self.model.evaluate(\
                            [val_x[self.sam_cols[i]] for i in \
                            range(self.net_width)],\
                            val_y,\
                            batch_size=batch_size)       

