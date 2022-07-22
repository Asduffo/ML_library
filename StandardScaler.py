# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 21:40:27 2020

@author: Ninniri Matteo (m.ninniri1@studenti.unipi.it)
"""
import numpy as np
from BaseModel import BaseTransformer

class StandardScaler(BaseTransformer):
    def __init__(self):
        super().__init__()
        
    def fit(self, X, y = None, X_vl = None, y_vl = None):
        super().fit(X, y, X_vl, y_vl)
        
        self.averages = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std = np.where(self.std == 0, 1, self.std)
    
    def transform(self, X, y = []):
        #TODO: check valid sizes
        
        X_final = X.copy()
        y_final = y.copy()
        
        X_final = X_final - self.averages
        # print(X_final)
        X_final = X_final / self.std
        
        return X_final, y_final
    
    def setup_hyperparam(self, hyperparam_name, hyperparam_value):
        raise Exception("StandardScaler has no hyperparameters")
        