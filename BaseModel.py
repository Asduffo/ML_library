# -*- coding: utf-8 -*-
"""
Classes implenting the basic units of our machine learning models

@author: Lepri Elena (e.lepri@studenti.unipi.it)
         Ninniri Matteo (m.ninniri1@studenti.unipi.it)
"""

from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    """ Abstract class which both transformers and models inherit from """
    
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def fit(self, X, y = None, X_vl = None, y_vl = None):
        """ Train the model/transformer on the training data.
        
        Parameters:
        X (np.ndarray): Training data.
        y (np.ndarray) optional: Training labels.
        X_vl (np.ndarray) optional: Validation data.
        y_vl (np.ndarray) optional: Validation labels.
        """
        pass
    
    @abstractmethod
    def setup_hyperparam(self, hyperparam_name, hyperparam_value):
        """ Set an hyperparameter identified by hyperparam_name to
            the value specified by hyperparam_value.
        
        Parameters:
        hyperparam_name (list of string): Path to the hyperparameter to be set.
        hyperparam_value (any): Value to assign to the hyperparameter
        """
        pass
    
class BaseTransformer(BaseModel):
    """ Abstract class which all transformers inherit from """
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def transform(self, X, y = None, X_vl = None, y_vl = None):
        """ Transform the data.
        
        Parameters:
        X (np.ndarray): Data to be transformed.
        y (np.ndarray) optional: Labels to be transformed.
        X_vl (np.ndarray) optional: Validation data to be transformed.
        y_vl (np.ndarray) optional: Validation labels to be transformed.
        """
        
        assert(isinstance(X, (np.ndarray))), "BaseTransformer.fit: X must be an ndarray!"
        
        assert(isinstance(y, (np.ndarray, type(None)))), "BaseTransformer.fit: y must be an ndarray or None!"
        assert(isinstance(X_vl, (np.ndarray, type(None)))), "BaseTransformer.fit: X_vl must be an ndarray!"
        assert(isinstance(y_vl, (np.ndarray, type(None)))), "BaseTransformer.fit: y_vl must be an ndarray!"
        if(y != None):
            assert(X.shape[0] == y.shape[0]), "BaseTransformer.fit: uncoherent dataset and label matrices rows"
        if(y_vl != None):
            assert(X_vl.shape[0] == y_vl.shape[0]), "BaseTransformer.fit: uncoherent dataset and label matrices rows for validation set"
        
    
class FinalModel(BaseModel):
    """ Abstract class which all models inherit from """
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def predict(self, X):
        """ Predict the output associated to the input data.
        
        Parameters:
        X (np.ndarray): Input data.
        """
        pass
    
    @abstractmethod
    def score(self, X, y):
        """ Calculate the accuracy metric between y and the output of X.
        
        Parameters:
        X (np.ndarray): Data which we want to generalize from.
        y (np.ndarray): Expected output of X.
        """
        pass